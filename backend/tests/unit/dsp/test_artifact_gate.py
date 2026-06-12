"""Unit tests for dsp/artifact_gate.py (Stage 3 epoch-level artifact gate)."""

from __future__ import annotations

import numpy as np
import pytest

from neurolink.dsp.artifact_gate import ArtifactGate, ArtifactDecision, GateConfig

FS = 256.0
N_SAMPLES = 256
N_CH = 5


def _clean_eeg(amplitude_uv: float = 20.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((N_CH, N_SAMPLES)) * amplitude_uv).astype(np.float32)


def _clean_accel(rms_g: float = 0.02) -> np.ndarray:
    """Stationary accel — tiny constant gravity-aligned vector."""
    rng = np.random.default_rng(99)
    a = rng.standard_normal((3, 20)) * rms_g
    return a.astype(np.float32)


# ---------------------------------------------------------------------------
# Clean frame — all gates pass
# ---------------------------------------------------------------------------

class TestCleanFrame:
    def test_clean_eeg_no_reject(self):
        gate = ArtifactGate(GateConfig(pk2pk_uv=100.0, accel_rms_g=0.15))
        eeg = _clean_eeg(amplitude_uv=10.0)  # ±10 µV noise → pk2pk ~40 µV
        decision = gate.evaluate(eeg, _clean_accel())
        assert decision.clean
        assert decision.reasons == []

    def test_clean_frame_counted_in_stats(self):
        gate = ArtifactGate()
        eeg = _clean_eeg(amplitude_uv=5.0)
        gate.evaluate(eeg, _clean_accel())
        stats = gate.get_stats()
        assert stats["total_frames"] == 1
        assert stats["rejected_frames"] == 0


# ---------------------------------------------------------------------------
# Amplitude gate
# ---------------------------------------------------------------------------

class TestAmplitudeGate:
    def test_high_amplitude_rejected(self):
        gate = ArtifactGate(GateConfig(pk2pk_uv=100.0, enable_imu=False, enable_kurtosis=False))
        eeg = _clean_eeg()
        # Inject a large spike on AF7 (index 1)
        eeg[1, 100] = 200.0
        eeg[1, 101] = -200.0  # peak-to-peak = 400 µV
        decision = gate.evaluate(eeg)
        assert decision.reject
        assert any("amplitude" in r for r in decision.reasons)

    def test_amplitude_gate_disabled(self):
        gate = ArtifactGate(
            GateConfig(pk2pk_uv=1.0, enable_amplitude=False, enable_imu=False, enable_kurtosis=False)
        )
        eeg = _clean_eeg(amplitude_uv=50.0)  # big but gate disabled
        decision = gate.evaluate(eeg)
        assert decision.clean

    def test_aux_channel_ignored_by_amplitude_gate(self):
        gate = ArtifactGate(GateConfig(pk2pk_uv=100.0, enable_imu=False, enable_kurtosis=False))
        eeg = _clean_eeg(amplitude_uv=5.0)
        # Only AUX (index 4) has high amplitude
        eeg[4, :] = 500.0
        eeg[4, 0] = -500.0
        decision = gate.evaluate(eeg)
        assert decision.clean  # AUX excluded from EEG checks


# ---------------------------------------------------------------------------
# IMU gate
# ---------------------------------------------------------------------------

class TestIMUGate:
    def test_motion_rejected(self):
        gate = ArtifactGate(
            GateConfig(accel_rms_g=0.15, enable_amplitude=False, enable_kurtosis=False)
        )
        # 1 g RMS — vigorous movement
        accel = np.ones((3, 20), dtype=np.float32)
        eeg = _clean_eeg()
        decision = gate.evaluate(eeg, accel)
        assert decision.reject
        assert any("imu" in r for r in decision.reasons)

    def test_no_accel_skips_imu_gate(self):
        gate = ArtifactGate(
            GateConfig(enable_amplitude=False, enable_kurtosis=False)
        )
        eeg = _clean_eeg()
        decision = gate.evaluate(eeg, accel=None)
        assert decision.clean

    def test_imu_gate_disabled(self):
        gate = ArtifactGate(
            GateConfig(accel_rms_g=0.0001, enable_imu=False, enable_amplitude=False, enable_kurtosis=False)
        )
        accel = np.ones((3, 20), dtype=np.float32) * 2.0
        eeg = _clean_eeg()
        decision = gate.evaluate(eeg, accel)
        assert decision.clean

    def test_1d_accel_handled(self):
        gate = ArtifactGate(
            GateConfig(accel_rms_g=0.15, enable_amplitude=False, enable_kurtosis=False)
        )
        accel_1d = np.ones(20, dtype=np.float32) * 0.01  # low — should pass
        eeg = _clean_eeg()
        decision = gate.evaluate(eeg, accel_1d)
        assert decision.clean


# ---------------------------------------------------------------------------
# Kurtosis gate
# ---------------------------------------------------------------------------

class TestKurtosisGate:
    def test_high_kurtosis_burst_rejected(self):
        gate = ArtifactGate(
            GateConfig(kurtosis_threshold=5.0, enable_amplitude=False, enable_imu=False)
        )
        eeg = _clean_eeg(amplitude_uv=5.0)  # baseline: near-Gaussian
        # Inject sharp burst on TP9 (index 0) — creates high kurtosis
        eeg[0] = 0.0
        eeg[0, 50] = 300.0   # single spike on flat background
        decision = gate.evaluate(eeg)
        assert decision.reject
        assert any("kurtosis" in r for r in decision.reasons)

    def test_gaussian_noise_passes_kurtosis(self):
        gate = ArtifactGate(
            GateConfig(kurtosis_threshold=5.0, enable_amplitude=False, enable_imu=False)
        )
        rng = np.random.default_rng(7)
        eeg = (rng.standard_normal((N_CH, 1024)) * 10.0).astype(np.float32)
        decision = gate.evaluate(eeg)
        # Gaussian kurtosis ≈ 0 (Fisher); should not exceed threshold of 5
        assert decision.clean

    def test_kurtosis_gate_disabled(self):
        gate = ArtifactGate(
            GateConfig(kurtosis_threshold=0.0, enable_kurtosis=False,
                       enable_amplitude=False, enable_imu=False)
        )
        eeg = _clean_eeg()
        decision = gate.evaluate(eeg)
        assert decision.clean


# ---------------------------------------------------------------------------
# Stats and config API
# ---------------------------------------------------------------------------

class TestStatsAndConfig:
    def test_rejection_rate_computed(self):
        gate = ArtifactGate(GateConfig(pk2pk_uv=1.0, enable_imu=False, enable_kurtosis=False))
        eeg_bad = _clean_eeg(amplitude_uv=50.0)  # will be rejected
        eeg_clean = np.zeros((N_CH, N_SAMPLES), dtype=np.float32)  # pk2pk=0
        gate.evaluate(eeg_bad)
        gate.evaluate(eeg_clean)
        stats = gate.get_stats()
        assert stats["total_frames"] == 2
        assert stats["rejected_frames"] == 1
        assert stats["rejection_rate"] == 0.5

    def test_reset_stats_clears_counters(self):
        gate = ArtifactGate(GateConfig(pk2pk_uv=1.0, enable_imu=False, enable_kurtosis=False))
        eeg = _clean_eeg(amplitude_uv=50.0)
        gate.evaluate(eeg)
        gate.reset_stats()
        stats = gate.get_stats()
        assert stats["total_frames"] == 0
        assert stats["rejected_frames"] == 0

    def test_set_config_updates_thresholds(self):
        gate = ArtifactGate()
        gate.set_config(GateConfig(pk2pk_uv=200.0, accel_rms_g=0.5))
        cfg = gate.get_config()
        assert cfg.pk2pk_uv == 200.0
        assert cfg.accel_rms_g == 0.5


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_none_eeg_returns_clean_decision(self):
        gate = ArtifactGate()
        decision = gate.evaluate(None)  # type: ignore[arg-type]
        assert decision.clean

    def test_single_sample_buffer_returns_clean(self):
        gate = ArtifactGate()
        eeg = np.zeros((N_CH, 1), dtype=np.float32)
        decision = gate.evaluate(eeg)
        assert decision.clean

    def test_1d_eeg_returns_clean(self):
        """1-D input is rejected early (requires ndim==2)."""
        gate = ArtifactGate()
        eeg = np.zeros(N_SAMPLES, dtype=np.float32)
        decision = gate.evaluate(eeg)  # type: ignore[arg-type]
        assert decision.clean
