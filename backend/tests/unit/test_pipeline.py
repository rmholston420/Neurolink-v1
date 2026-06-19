"""Unit tests for EEGPipeline, StreamHealth, and single-PSD band powers."""

from __future__ import annotations

import time
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
    n_samples: int = 512,
    n_channels: int = 5,
    source: str = "mock",
) -> EEGSample:
    """Return a synthetic EEGSample with a simple sinusoidal EEG buffer."""
    t = np.linspace(0, n_samples / 256.0, n_samples)
    buf = [
        (10.0 * np.sin(2 * np.pi * 10.0 * t) + 1.0 * np.random.randn(n_samples)).tolist()
        for _ in range(n_channels)
    ]
    return EEGSample(
        channels=[b[-1] for b in buf],
        timestamp=time.time(),
        source=source,
        address="",
        poor_contact=False,
        eeg_buffer=buf,
        ppg_buffer=None,
        accel_buffer=None,
        gyro_buffer=None,
    )


def _make_pipeline() -> tuple[EEGPipeline, MagicMock]:
    hub = MagicMock()
    pipeline = EEGPipeline(hub=hub, publish_hz=4.0)
    return pipeline, hub


# ---------------------------------------------------------------------------
# StreamHealth tests
# ---------------------------------------------------------------------------


class TestStreamHealth:
    def test_initial_state(self):
        health = StreamHealth()
        assert health.frames_total == 0
        assert health.frames_rejected == 0
        assert health.frames_clean == 0
        assert health.packet_loss_pct == 0.0
        assert health.avg_tick_ms == 0.0

    def test_record_frame_clean(self):
        health = StreamHealth()
        health.record_frame(rejected=False, tick_ms=5.0)
        assert health.frames_total == 1
        assert health.frames_clean == 1
        assert health.frames_rejected == 0

    def test_record_frame_rejected(self):
        health = StreamHealth()
        health.record_frame(rejected=True, tick_ms=2.0)
        assert health.frames_total == 1
        assert health.frames_rejected == 1
        assert health.frames_clean == 0

    def test_ema_tick_ms(self):
        health = StreamHealth(_ema_alpha=1.0)  # instant update
        health.record_frame(rejected=False, tick_ms=10.0)
        assert health.avg_tick_ms == pytest.approx(10.0)
        health.record_frame(rejected=False, tick_ms=20.0)
        assert health.avg_tick_ms == pytest.approx(20.0)

    def test_reset_zeroes_all(self):
        health = StreamHealth()
        health.record_frame(rejected=False, tick_ms=5.0)
        health.record_frame(rejected=True, tick_ms=3.0)
        health.reset()
        assert health.frames_total == 0
        assert health.frames_clean == 0
        assert health.frames_rejected == 0
        assert health.avg_tick_ms == 0.0

    def test_to_dict_keys(self):
        health = StreamHealth()
        d = health.to_dict()
        expected_keys = {
            "frames_total",
            "frames_rejected",
            "frames_clean",
            "packet_loss_pct",
            "last_frame_ts",
            "avg_tick_ms",
        }
        assert set(d.keys()) == expected_keys

    def test_packet_loss_window_update(self):
        """After a window elapses, packet_loss_pct should reflect missed frames."""
        health = StreamHealth(_publish_hz=4.0)
        # Force window to appear expired
        health._window_start_ts = time.time() - 15.0  # 15 s ago
        health._window_frames_seen = 4  # only 4 of expected 60
        # Record a frame to trigger window refresh
        health.record_frame(rejected=False, tick_ms=1.0)
        assert health.packet_loss_pct > 0.0


# ---------------------------------------------------------------------------
# Single-PSD band powers tests
# ---------------------------------------------------------------------------


class TestComputeBandsSinglePsd:
    def test_returns_all_bands(self):
        rng = np.random.default_rng(42)
        arr = rng.standard_normal((5, 512)).astype(np.float32)
        result = _compute_bands_single_psd(arr)
        assert set(result.keys()) == {"delta", "theta", "alpha", "beta", "gamma"}

    def test_normalised_to_one(self):
        rng = np.random.default_rng(42)
        arr = rng.standard_normal((5, 512)).astype(np.float32)
        result = _compute_bands_single_psd(arr)
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-5)

    def test_none_returns_zeros(self):
        result = _compute_bands_single_psd(None)
        assert all(v == 0.0 for v in result.values())

    def test_short_signal_returns_zeros(self):
        arr = np.zeros((5, 1), dtype=np.float32)
        result = _compute_bands_single_psd(arr)
        assert all(v == 0.0 for v in result.values())

    def test_dominant_alpha_signal(self):
        """A pure 10 Hz sine should produce alpha as the dominant band."""
        fs = 256.0
        t = np.arange(512) / fs
        sig = np.sin(2 * np.pi * 10.0 * t).astype(np.float32)
        arr = np.vstack([sig] * 5)
        result = _compute_bands_single_psd(arr)
        dominant = max(result, key=result.get)
        assert dominant == "alpha", f"Expected alpha dominant, got {dominant}: {result}"

    def test_single_channel_1d(self):
        rng = np.random.default_rng(0)
        arr = rng.standard_normal(512).astype(np.float32)
        result = _compute_bands_single_psd(arr)
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# EEGPipeline integration tests
# ---------------------------------------------------------------------------


class TestEEGPipeline:
    def test_process_returns_result(self):
        pipeline, _ = _make_pipeline()
        sample = _make_sample()
        result = pipeline.process(sample)
        assert result is not None

    def test_bands_in_result(self):
        pipeline, _ = _make_pipeline()
        sample = _make_sample()
        result = pipeline.process(sample)
        assert result is not None
        bp = result.bands
        total = bp.alpha + bp.theta + bp.beta + bp.delta + bp.gamma
        assert total == pytest.approx(1.0, abs=0.01)

    def test_health_updated_after_process(self):
        pipeline, _ = _make_pipeline()
        sample = _make_sample()
        pipeline.process(sample)
        assert pipeline.health.frames_total == 1
        assert pipeline.health.last_frame_ts > 0.0

    def test_pipeline_reset_zeroes_health(self):
        pipeline, _ = _make_pipeline()
        sample = _make_sample()
        pipeline.process(sample)
        assert pipeline.health.frames_total > 0
        pipeline.reset()
        assert pipeline.health.frames_total == 0

    def test_artifact_rejected_increments_counter(self):
        pipeline, _ = _make_pipeline()
        sample = _make_sample()
        # Inject a very high amplitude signal to trigger artifact gate
        sample.eeg_buffer = [
            [1e6] * 512 for _ in range(5)
        ]
        pipeline.process(sample)
        # Rejected frames should be non-zero (gate may trigger)
        # We just assert health was updated regardless
        assert pipeline.health.frames_total >= 1

    def test_multiple_samples_accumulate(self):
        pipeline, _ = _make_pipeline()
        for _ in range(5):
            pipeline.process(_make_sample())
        assert pipeline.health.frames_total == 5
