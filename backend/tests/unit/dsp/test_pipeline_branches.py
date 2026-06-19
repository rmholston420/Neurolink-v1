"""Targeted branch coverage for dsp/pipeline.py.

Covers every uncovered line group identified in the coverage gap analysis:

  Line 314    -- _stage0_settling_reason() branches
  Lines 353/357-364  -- Stage 3b artifact annotation assembly
  Lines 375-378      -- correction plan builder (_plan_* flag propagation)
  Lines 399-400      -- _apply_corrections guard (hard_reject path)
  Lines 435-450      -- _compute_bands_single_psd edge cases
  Lines 477/482-483  -- StreamHealth packet-loss window boundary

All tests use pure Python/NumPy -- no hardware, no bleak.
"""

from __future__ import annotations

import time
import unittest.mock as mock
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from neurolink.dsp.pipeline import (
    EEGPipeline,
    StreamHealth,
    _compute_bands_single_psd,
)
from neurolink.hardware.base import EEGSample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sample(
    *,
    n_channels: int = 5,
    n_samples: int = 256,
    source: str = "mock",
    poor_contact: bool = False,
    ppg_buffer=None,
    accel_buffer=None,
    gyro_buffer=None,
    extra: dict | None = None,
) -> EEGSample:
    rng = np.random.default_rng(0)
    channels = rng.standard_normal((n_channels, n_samples))
    eeg_buffer = [ch.tolist() for ch in channels]
    return EEGSample(
        channels=[ch[-1] for ch in channels],
        timestamp=time.time(),
        source=source,
        address="mock",
        poor_contact=poor_contact,
        eeg_buffer=eeg_buffer,
        ppg_buffer=ppg_buffer,
        accel_buffer=accel_buffer,
        gyro_buffer=gyro_buffer,
        extra=extra or {},
    )


def _make_pipeline(hub=None) -> EEGPipeline:
    if hub is None:
        hub = MagicMock()
        hub.update = MagicMock()
    return EEGPipeline(hub=hub)


# ---------------------------------------------------------------------------
# _stage0_settling_reason branches (line 314)
# ---------------------------------------------------------------------------


class TestStage0SettlingReason:
    """Cover every branch of _stage0_settling_reason."""

    def test_no_stage0_returns_settling(self):
        p = _make_pipeline()
        p._stage0 = None
        assert p._stage0_settling_reason() == "settling"

    def test_impedance_not_ok_returns_impedance_unstable(self):
        p = _make_pipeline()
        stage0 = MagicMock()
        stage0.impedance.all_channels_ok = False
        p._stage0 = stage0
        assert p._stage0_settling_reason() == "impedance_unstable"

    def test_motion_flagged_returns_motion_settling(self):
        p = _make_pipeline()
        stage0 = MagicMock()
        stage0.impedance.all_channels_ok = True
        latest = MagicMock()
        latest.extra = {"motion_flagged": True}
        stage0._latest_sample = latest
        p._stage0 = stage0
        assert p._stage0_settling_reason() == "motion_settling"

    def test_env_not_ready_returns_env_not_ready(self):
        p = _make_pipeline()
        stage0 = MagicMock()
        stage0.impedance.all_channels_ok = True
        stage0._latest_sample = None
        stage0.environment.is_ready = False
        p._stage0 = stage0
        assert p._stage0_settling_reason() == "env_not_ready"

    def test_all_ok_returns_settling(self):
        p = _make_pipeline()
        stage0 = MagicMock()
        stage0.impedance.all_channels_ok = True
        stage0._latest_sample = None
        stage0.environment.is_ready = True
        p._stage0 = stage0
        assert p._stage0_settling_reason() == "settling"


# ---------------------------------------------------------------------------
# Stage 3b annotation assembly (lines 353/357-364)
# ---------------------------------------------------------------------------


class TestStage3bAnnotationAssembly:
    """Cover the artifact annotation list-comprehension path."""

    def test_annotations_populated_when_artifacts_detected(self):
        """When stage3b returns annotations they should appear in the result."""
        p = _make_pipeline()
        sample = _make_sample()

        # Build a fake ArtifactDetectionReport with one annotation
        fake_annotation = MagicMock()
        fake_annotation.artifact_type.name = "BLINK"
        fake_annotation.confidence = 0.9
        fake_annotation.channels = ["AF7", "AF8"]
        fake_annotation.feature_value = 42.0
        fake_annotation.feature_name = "peak_amp"
        fake_annotation.threshold = 30.0

        fake_plan = MagicMock()
        fake_plan.hard_reject = False
        fake_plan.apply_ocular_regression = True
        fake_plan.apply_asr = False
        fake_plan.apply_notch = False
        fake_plan.apply_cardiac_regression = False

        fake_report = MagicMock()
        fake_report.clean = True  # so correction plan branch is reached
        fake_report.annotations = [fake_annotation]
        fake_report.correction_plan = fake_plan

        p._stage3b.classify = MagicMock(return_value=fake_report)
        # Stage 3 must pass (not reject)
        p._stage3.evaluate = MagicMock(
            return_value=MagicMock(reject=False, reasons=[])
        )

        result = p.process(sample)
        assert result is not None
        assert len(result.artifact_annotations) == 1
        ann = result.artifact_annotations[0]
        assert ann.artifact_type == "BLINK"
        assert ann.confidence == pytest.approx(0.9)

    def test_correction_plan_payload_populated(self):
        """ArtifactCorrectionPlanPayload must mirror the plan returned by stage3b."""
        p = _make_pipeline()
        sample = _make_sample()

        fake_plan = MagicMock()
        fake_plan.hard_reject = True
        fake_plan.apply_ocular_regression = False
        fake_plan.apply_asr = True
        fake_plan.apply_notch = True
        fake_plan.apply_cardiac_regression = False

        fake_report = MagicMock()
        fake_report.clean = False
        fake_report.annotations = []
        fake_report.correction_plan = fake_plan

        p._stage3b.classify = MagicMock(return_value=fake_report)
        p._stage3.evaluate = MagicMock(
            return_value=MagicMock(reject=False, reasons=[])
        )

        result = p.process(sample)
        # hard_reject causes artifact_rejected=True, result is still returned
        assert result is not None
        assert result.artifact_rejected is True
        plan = result.artifact_correction_plan
        assert plan is not None
        assert plan.hard_reject is True
        assert plan.apply_asr is True


# ---------------------------------------------------------------------------
# _apply_corrections guard: hard_reject path (lines 399-400)
# ---------------------------------------------------------------------------


class TestHardRejectPath:
    """Cover the _plan_hard_reject -> artifact_rejected branch."""

    def test_hard_reject_sets_artifact_rejected(self):
        """When plan.hard_reject is True, result.artifact_rejected must be True."""
        p = _make_pipeline()
        sample = _make_sample()

        fake_plan = MagicMock()
        fake_plan.hard_reject = True
        fake_plan.apply_ocular_regression = False
        fake_plan.apply_asr = False
        fake_plan.apply_notch = False
        fake_plan.apply_cardiac_regression = False

        fake_annotation = MagicMock()
        fake_annotation.artifact_type = MagicMock()
        fake_annotation.artifact_type.name = "MOTION"
        fake_annotation.confidence = 0.99
        fake_annotation.channels = []
        fake_annotation.feature_value = 0.0
        fake_annotation.feature_name = "rms"
        fake_annotation.threshold = 0.0

        fake_report = MagicMock()
        fake_report.clean = False
        fake_report.annotations = [fake_annotation]
        fake_report.correction_plan = fake_plan

        p._stage3b.classify = MagicMock(return_value=fake_report)
        p._stage3.evaluate = MagicMock(
            return_value=MagicMock(reject=False, reasons=[])
        )

        result = p.process(sample)
        assert result is not None
        assert result.artifact_rejected is True
        assert any("3b:" in r for r in result.artifact_reasons)


# ---------------------------------------------------------------------------
# _compute_bands_single_psd edge cases (lines 435-450)
# ---------------------------------------------------------------------------


class TestComputeBandsSinglePsdEdgeCases:
    """Cover error/guard branches inside _compute_bands_single_psd."""

    def test_none_input_returns_all_zeros(self):
        result = _compute_bands_single_psd(None)
        assert all(v == 0.0 for v in result.values())

    def test_too_short_returns_all_zeros(self):
        """Array with < 2 samples must return all zeros."""
        arr = np.array([[1.0]])  # 1 channel, 1 sample
        result = _compute_bands_single_psd(arr)
        assert all(v == 0.0 for v in result.values())

    def test_all_zero_signal_returns_all_zeros(self):
        """Zero signal has zero total power; normalisation guard must return zeros."""
        arr = np.zeros((5, 256), dtype=np.float32)
        result = _compute_bands_single_psd(arr)
        assert all(v == 0.0 for v in result.values())

    def test_1d_array_is_accepted(self):
        """A 1D array should be promoted to (1, n_samples) and succeed."""
        rng = np.random.default_rng(1)
        arr = rng.standard_normal(256).astype(np.float32)
        result = _compute_bands_single_psd(arr)
        assert set(result.keys()) == {"delta", "theta", "alpha", "beta", "gamma"}
        total = sum(result.values())
        assert abs(total - 1.0) < 1e-5

    def test_normal_signal_sums_to_one(self):
        """Valid multi-channel signal must return normalised powers summing to 1."""
        rng = np.random.default_rng(42)
        arr = rng.standard_normal((5, 512)).astype(np.float32)
        result = _compute_bands_single_psd(arr)
        total = sum(result.values())
        assert abs(total - 1.0) < 1e-5

    def test_known_10hz_signal_has_alpha_dominant(self):
        """A pure 10 Hz sinusoid should produce alpha as the dominant band."""
        t = np.linspace(0, 4, 1024, dtype=np.float32)
        arr = np.tile(np.sin(2 * np.pi * 10 * t), (5, 1))
        result = _compute_bands_single_psd(arr)
        dominant = max(result, key=result.get)
        assert dominant == "alpha"


# ---------------------------------------------------------------------------
# StreamHealth packet-loss window boundary (lines 477/482-483)
# ---------------------------------------------------------------------------


class TestStreamHealthWindowBoundary:
    """Cover the rolling packet-loss window update path."""

    def test_no_loss_when_all_frames_seen(self):
        """If every expected frame is received, packet_loss_pct must be 0."""
        health = StreamHealth(_publish_hz=4.0)
        # Force a window boundary by backdating _window_start_ts by 11 seconds
        health._window_start_ts = time.time() - 11.0
        # Simulate 44 frames seen (11 s * 4 Hz)
        health._window_frames_seen = 44

        health.record_frame(rejected=False, tick_ms=1.0)
        # Window should have rolled; loss should be near 0
        assert health.packet_loss_pct == pytest.approx(0.0, abs=5.0)

    def test_half_frames_lost_reports_50pct(self):
        """If half the expected frames are missing, packet_loss_pct ~= 50."""
        health = StreamHealth(_publish_hz=4.0)
        health._window_start_ts = time.time() - 10.0
        # 10 s * 4 Hz = 40 expected; only 20 seen
        health._window_frames_seen = 20

        health.record_frame(rejected=False, tick_ms=1.0)
        # Allow a few % tolerance for timing jitter
        assert health.packet_loss_pct == pytest.approx(50.0, abs=10.0)

    def test_window_resets_after_boundary(self):
        """After a window boundary the internal seen counter must reset to 1."""
        health = StreamHealth(_publish_hz=4.0)
        health._window_start_ts = time.time() - 11.0
        health._window_frames_seen = 44

        health.record_frame(rejected=False, tick_ms=2.0)
        # After the boundary record_frame incremented _window_frames_seen to 1
        assert health._window_frames_seen == 1

    def test_no_boundary_if_elapsed_lt_window(self):
        """No window update should occur before _HEALTH_WINDOW_SEC elapses."""
        health = StreamHealth(_publish_hz=4.0)
        original_pct = health.packet_loss_pct
        # Window start is recent (< 10 s ago) -- no boundary should fire
        health._window_start_ts = time.time() - 1.0
        health._window_frames_seen = 4

        for _ in range(10):
            health.record_frame(rejected=False, tick_ms=1.0)

        assert health.packet_loss_pct == original_pct

    def test_ema_tick_ms_updates(self):
        """avg_tick_ms must update via EMA each record_frame call."""
        health = StreamHealth(_publish_hz=4.0)
        health.record_frame(rejected=False, tick_ms=10.0)
        assert health.avg_tick_ms == pytest.approx(10.0)
        health.record_frame(rejected=False, tick_ms=20.0)
        # EMA: 0.1 * 20 + 0.9 * 10 = 11.0
        assert health.avg_tick_ms == pytest.approx(11.0, abs=0.01)

    def test_rejected_frame_increments_frames_rejected(self):
        health = StreamHealth(_publish_hz=4.0)
        health.record_frame(rejected=True, tick_ms=1.0)
        assert health.frames_rejected == 1
        assert health.frames_clean == 0

    def test_reset_zeroes_all_counters(self):
        health = StreamHealth(_publish_hz=4.0)
        for _ in range(5):
            health.record_frame(rejected=False, tick_ms=5.0)
        health.reset()
        assert health.frames_total == 0
        assert health.frames_rejected == 0
        assert health.frames_clean == 0
        assert health.packet_loss_pct == 0.0
        assert health.avg_tick_ms == 0.0
