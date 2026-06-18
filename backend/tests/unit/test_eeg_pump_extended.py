"""Targeted tests for EEGPump missing lines.

Missing lines to cover:
  117  - start() idempotency (_running already True)
  135  - stop() when _task is None
  145  - reset() clears baseline/stage6/hub
  155  - _pump_loop watchdog log (no frames for > _WATCHDOG_SEC)
  192  - _stage0_settling_reason: impedance_unstable branch
  302-311 - _tick: stage0 gate_sample + impedance update
  328-330 - _tick: stage0 acquisition_ready=False, source != 'mock' -> emit_settling + return
  339-341 - _tick: _last_frame_ts set, set_latest_sample, build_payload called
  359    - _build_payload: stage1 FIR applied when toggle on
  372-373 - _build_payload: stage2 bad channels -> spherical spline interpolation
  435-446 - _build_payload: PPG path (ppg_buffer present)
  449-451 - _build_payload: stage6 cardiac regression with ibis
  480    - _build_payload: IMU payload when accel+gyro present
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import numpy as np
import pytest

from neurolink.eeg_pump import EEGPump, _wire_stubs, _MIN_PPG_SAMPLES
from neurolink.hub import EEGHub
from neurolink.hardware.base import EEGSample
from neurolink.dsp.filter_toggles import FilterToggleConfig


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_eeg_sample(
    source: str = "mock",
    eeg: bool = True,
    ppg: bool = False,
    accel: bool = False,
    gyro: bool = False,
    poor_contact: bool = False,
) -> MagicMock:
    sample = MagicMock(spec=EEGSample)
    sample.source = source
    sample.address = "00:00:00:00"
    sample.timestamp = 0.0
    sample.poor_contact = poor_contact
    sample.channels = None
    sample.extra = {}

    if eeg:
        t = np.linspace(0, 1, 64)
        ch = (np.sin(2 * np.pi * 10 * t) * 0.4).tolist()
        sample.eeg_buffer = [ch for _ in range(4)]
    else:
        sample.eeg_buffer = None

    # PPG buffer must be >= _MIN_PPG_SAMPLES (960) to clear the warmup guard
    # introduced in eeg_pump.py.  A shorter buffer causes ppg_payload=None
    # which is the correct runtime behaviour during warmup, but tests that
    # want to exercise the PPG code path need a full-length buffer.
    sample.ppg_buffer = ([0.5] * _MIN_PPG_SAMPLES if ppg else None)
    sample.accel_buffer = ([[0.01] * 10, [0.01] * 10, [1.0] * 10] if accel else None)
    sample.gyro_buffer = ([[0.0] * 10, [0.0] * 10, [0.0] * 10] if gyro else None)
    return sample


def _make_pump(adapter=None, hub=None, stage0=None):
    if adapter is None:
        adapter = MagicMock()
        adapter.read_sample = AsyncMock(return_value=_make_eeg_sample())
    if hub is None:
        hub = EEGHub()
    return EEGPump(adapter=adapter, hub=hub, stage0_guard=stage0)


def _all_toggles_off() -> FilterToggleConfig:
    return FilterToggleConfig(
        stage1_fir=False,
        stage2_bad_channels=False,
        stage3_artifact_gate=False,
        stage3b_artifact_detector=False,
        stage4_asr=False,
        stage4b_baseline=False,
        stage5_ocular=False,
        stage6_cardiac=False,
        imu_gate=False,
    )


# ═════════════════════════════════════════════════════════════════════════════
# start / stop / reset lifecycle  (lines 117, 135, 145)
# ═════════════════════════════════════════════════════════════════════════════

class TestEEGPumpLifecycle:
    @pytest.mark.asyncio
    async def test_start_idempotent_when_already_running(self):
        """Line 117: start() returns early if _running is already True."""
        pump = _make_pump()
        pump._running = True
        # No task should be created
        original_task = pump._task
        await pump.start()
        assert pump._task is original_task
        pump._running = False

    @pytest.mark.asyncio
    async def test_stop_with_no_task_is_safe(self):
        """Line 135: stop() when _task is None should not raise."""
        pump = _make_pump()
        pump._running = True
        pump._task = None
        await pump.stop()
        assert pump._running is False

    @pytest.mark.asyncio
    async def test_reset_clears_state(self):
        """Line 145: reset() calls baseline.reset(), stage6.reset(), hub.reset()."""
        hub = MagicMock()
        pump = _make_pump(hub=hub)
        pump._baseline = MagicMock()
        pump._stage6 = MagicMock()
        pump.reset()
        pump._baseline.reset.assert_called_once()
        pump._stage6.reset.assert_called_once()
        hub.reset.assert_called_once()


# ═════════════════════════════════════════════════════════════════════════════
# _pump_loop watchdog  (line 155)
# ═════════════════════════════════════════════════════════════════════════════

class TestEEGPumpWatchdog:
    @pytest.mark.asyncio
    async def test_watchdog_logs_when_no_frames(self):
        """Line 155: watchdog warning fires when _last_frame_ts is stale."""
        adapter = MagicMock()
        # Tick raises to force loop exit after one iteration
        adapter.read_sample = AsyncMock(side_effect=asyncio.CancelledError)
        pump = _make_pump(adapter=adapter)
        pump._running = True
        # Backdate _last_frame_ts by more than _WATCHDOG_SEC (10s)
        import time
        pump._last_frame_ts = time.time() - 15.0

        with patch("neurolink.eeg_pump.log") as mock_log:
            pump._running = False  # single iteration
            # Run one pass of the loop body directly
            pump._running = True
            tick_calls = 0

            async def _one_tick():
                nonlocal tick_calls
                tick_calls += 1
                pump._running = False  # stop after first

            pump._tick = _one_tick
            await pump._pump_loop()
            mock_log.warning.assert_called_with(
                "eeg_pump_no_frames", since_sec=10.0
            )


# ═════════════════════════════════════════════════════════════════════════════
# _stage0_settling_reason  (line 192)
# ═════════════════════════════════════════════════════════════════════════════

class TestStage0SettlingReason:
    def test_no_stage0_returns_settling(self):
        pump = _make_pump(stage0=None)
        assert pump._stage0_settling_reason() == "settling"

    def test_impedance_unstable_reason(self):
        stage0 = MagicMock()
        stage0.impedance.all_channels_ok = False
        pump = _make_pump(stage0=stage0)
        assert pump._stage0_settling_reason() == "impedance_unstable"

    def test_motion_settling_reason(self):
        stage0 = MagicMock()
        stage0.impedance.all_channels_ok = True
        sample = MagicMock()
        sample.extra = {"motion_flagged": True}
        stage0._latest_sample = sample
        stage0.environment.is_ready = True
        pump = _make_pump(stage0=stage0)
        assert pump._stage0_settling_reason() == "motion_settling"

    def test_env_not_ready_reason(self):
        stage0 = MagicMock()
        stage0.impedance.all_channels_ok = True
        stage0._latest_sample = None
        stage0.environment.is_ready = False
        pump = _make_pump(stage0=stage0)
        assert pump._stage0_settling_reason() == "env_not_ready"

    def test_all_ok_returns_settling(self):
        stage0 = MagicMock()
        stage0.impedance.all_channels_ok = True
        stage0._latest_sample = None
        stage0.environment.is_ready = True
        pump = _make_pump(stage0=stage0)
        assert pump._stage0_settling_reason() == "settling"


# ═════════════════════════════════════════════════════════════════════════════
# _tick with stage0  (lines 302-311, 328-341)
# ═════════════════════════════════════════════════════════════════════════════

class TestTickStage0:
    @pytest.mark.asyncio
    async def test_tick_calls_gate_sample_when_imu_toggle_on(self):
        """Lines 302-311: stage0.gate_sample and impedance.update_from_sample called."""
        sample = _make_eeg_sample(accel=True)
        adapter = MagicMock()
        adapter.read_sample = AsyncMock(return_value=sample)

        stage0 = MagicMock()
        stage0.gate_sample = MagicMock(return_value=sample)
        stage0.impedance.all_channels_ok = True
        stage0.acquisition_ready = True

        toggles = _all_toggles_off()
        toggles = FilterToggleConfig(**{**toggles.__dict__, "imu_gate": True})

        pump = _make_pump(adapter=adapter, stage0=stage0)

        with patch("neurolink.eeg_pump.filter_toggles") as ft, \
             patch("neurolink.eeg_pump.impedance") as imp:
            ft.get_toggles.return_value = toggles
            imp.check.return_value = True
            await pump._tick()

        stage0.gate_sample.assert_called_once_with(sample)
        stage0.impedance.update_from_sample.assert_called_once()

    @pytest.mark.asyncio
    async def test_tick_emits_settling_when_not_acquisition_ready(self):
        """Lines 328-330: non-mock source + not acquisition_ready -> emit_settling + early return."""
        sample = _make_eeg_sample(source="ble")
        adapter = MagicMock()
        adapter.read_sample = AsyncMock(return_value=sample)

        hub = MagicMock()
        hub.emit_settling = MagicMock()
        hub.set_latest_sample = MagicMock()
        hub.update = MagicMock()

        stage0 = MagicMock()
        stage0.gate_sample = MagicMock(return_value=sample)
        stage0.impedance.all_channels_ok = True
        stage0.acquisition_ready = False

        toggles = _all_toggles_off()
        pump = EEGPump(adapter=adapter, hub=hub, stage0_guard=stage0)

        with patch("neurolink.eeg_pump.filter_toggles") as ft, \
             patch("neurolink.eeg_pump.impedance") as imp:
            ft.get_toggles.return_value = toggles
            imp.check.return_value = True
            await pump._tick()

        hub.emit_settling.assert_called()
        hub.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_tick_sets_latest_sample_and_calls_update(self):
        """Lines 339-341: _last_frame_ts set, set_latest_sample, hub.update called."""
        sample = _make_eeg_sample()
        adapter = MagicMock()
        adapter.read_sample = AsyncMock(return_value=sample)

        hub = MagicMock()
        hub.update = MagicMock()
        hub.set_latest_sample = MagicMock()
        hub.emit_settling = MagicMock()
        hub.baseline_alpha = None

        toggles = _all_toggles_off()
        pump = EEGPump(adapter=adapter, hub=hub)
        pump._baseline = MagicMock()
        pump._baseline.process = MagicMock(side_effect=lambda x: x)
        pump._baseline.phase = "idle"

        with patch("neurolink.eeg_pump.filter_toggles") as ft, \
             patch("neurolink.eeg_pump.impedance") as imp:
            ft.get_toggles.return_value = toggles
            imp.check.return_value = True
            await pump._tick()

        hub.set_latest_sample.assert_called_once_with(sample)
        hub.update.assert_called_once()
        assert pump._last_frame_ts > 0


# ═════════════════════════════════════════════════════════════════════════════
# _build_payload DSP stage paths  (lines 359, 372-373, 435-451, 480)
# ═════════════════════════════════════════════════════════════════════════════

class TestBuildPayloadDSPPaths:
    @pytest.mark.asyncio
    async def test_stage1_fir_applied_when_toggle_on(self):
        """Line 359: stage1 FIR filter applied when toggle.stage1_fir=True."""
        sample = _make_eeg_sample()
        pump = _make_pump()
        pump._stage1 = MagicMock()
        pump._stage1.apply = MagicMock(side_effect=lambda x: x)

        toggles = _all_toggles_off()
        toggles = FilterToggleConfig(**{**toggles.__dict__, "stage1_fir": True})

        with patch("neurolink.eeg_pump.filter_toggles") as ft:
            ft.get_toggles.return_value = toggles
            await pump._build_payload(sample)

        pump._stage1.apply.assert_called()

    @pytest.mark.asyncio
    async def test_stage2_bad_channels_triggers_interpolation(self):
        """Lines 372-373: bad channel list -> spherical_spline.interpolate called."""
        sample = _make_eeg_sample()
        pump = _make_pump()

        toggles = FilterToggleConfig(**{**_all_toggles_off().__dict__, "stage2_bad_channels": True})

        with patch("neurolink.eeg_pump.filter_toggles") as ft, \
             patch("neurolink.eeg_pump.bad_channels") as bc, \
             patch("neurolink.eeg_pump.spherical_spline") as ss:
            ft.get_toggles.return_value = toggles
            bc.detect.return_value = ["TP9"]  # non-empty triggers interpolation
            ss.interpolate.return_value = np.zeros((4, 64), dtype=np.float32)
            payload = await pump._build_payload(sample)

        ss.interpolate.assert_called_once()
        assert "TP9" in payload.bad_channels

    @pytest.mark.asyncio
    async def test_ppg_buffer_produces_ppg_payload(self):
        """Lines 435-446: ppg_buffer >= _MIN_PPG_SAMPLES -> compute_ppg called, ppg not None.

        compute_ppg is patched so this test does not depend on neurokit2
        peak detection succeeding with synthetic data.
        """
        sample = _make_eeg_sample(ppg=True)  # buffer is now _MIN_PPG_SAMPLES long
        pump = _make_pump()

        toggles = _all_toggles_off()

        from neurolink.dsp.ppg import PPGPayload
        fake_ppg = PPGPayload(hr_bpm=60.0, ibi_ms=[833.0], sdnn_ms=20.0, rmssd_ms=18.0)

        with patch("neurolink.eeg_pump.filter_toggles") as ft, \
             patch("neurolink.eeg_pump._build_payload.__globals__", {}) as _dummy, \
             patch("neurolink.dsp.ppg.compute_ppg", return_value=fake_ppg), \
             patch("neurolink.eeg_pump.compute_ppg" if hasattr(__import__("neurolink.eeg_pump", fromlist=["compute_ppg"]), "compute_ppg") else "neurolink.dsp.ppg.compute_ppg", return_value=fake_ppg, create=True):
            ft.get_toggles.return_value = toggles
            # Patch the local import of compute_ppg inside _build_payload
            import neurolink.dsp.ppg as _ppg_mod
            orig = _ppg_mod.compute_ppg
            _ppg_mod.compute_ppg = lambda arr, fs: fake_ppg
            try:
                payload = await pump._build_payload(sample)
            finally:
                _ppg_mod.compute_ppg = orig

        assert payload.ppg is not None

    @pytest.mark.asyncio
    async def test_cardiac_regression_called_with_ibis(self):
        """Lines 449-451: stage6 cardiac regression runs when ppg+ibis present."""
        sample = _make_eeg_sample(ppg=True)  # buffer is now _MIN_PPG_SAMPLES long
        pump = _make_pump()

        toggles = FilterToggleConfig(**{**_all_toggles_off().__dict__, "stage6_cardiac": True})

        from neurolink.dsp.ppg import PPGPayload
        fake_ppg = PPGPayload(hr_bpm=60.0, ibi_ms=[833.0], sdnn_ms=20.0, rmssd_ms=18.0)

        import neurolink.dsp.ppg as _ppg_mod
        orig = _ppg_mod.compute_ppg
        _ppg_mod.compute_ppg = lambda arr, fs: fake_ppg
        try:
            with patch("neurolink.eeg_pump.filter_toggles") as ft, \
                 patch("neurolink.eeg_pump.cardiac_regression") as cr:
                ft.get_toggles.return_value = toggles
                cr.apply.return_value = np.zeros((4, 64), dtype=np.float32)
                await pump._build_payload(sample)
        finally:
            _ppg_mod.compute_ppg = orig

        cr.apply.assert_called_once()
        _, kwargs = cr.apply.call_args
        assert "ibis" in kwargs

    @pytest.mark.asyncio
    async def test_imu_payload_built_when_accel_and_gyro_present(self):
        """Line 480: IMU payload built when both accel_buffer and gyro_buffer present."""
        sample = _make_eeg_sample(accel=True, gyro=True)
        pump = _make_pump()

        toggles = _all_toggles_off()

        with patch("neurolink.eeg_pump.filter_toggles") as ft:
            ft.get_toggles.return_value = toggles
            payload = await pump._build_payload(sample)

        assert payload.imu is not None

    @pytest.mark.asyncio
    async def test_impedance_bad_emits_settling_in_tick(self):
        """Lines 302-305: impedance.check()=False -> hub.emit_settling called."""
        sample = _make_eeg_sample()
        adapter = MagicMock()
        adapter.read_sample = AsyncMock(return_value=sample)
        hub = MagicMock()
        hub.update = MagicMock()
        hub.set_latest_sample = MagicMock()
        hub.emit_settling = MagicMock()
        hub.baseline_alpha = None

        pump = EEGPump(adapter=adapter, hub=hub)
        pump._baseline = MagicMock()
        pump._baseline.process = MagicMock(side_effect=lambda x: x)
        pump._baseline.phase = "idle"

        toggles = _all_toggles_off()

        with patch("neurolink.eeg_pump.filter_toggles") as ft, \
             patch("neurolink.eeg_pump.impedance") as imp:
            ft.get_toggles.return_value = toggles
            imp.check.return_value = False  # bad impedance
            await pump._tick()

        hub.emit_settling.assert_called_with(reason="impedance_unstable")
