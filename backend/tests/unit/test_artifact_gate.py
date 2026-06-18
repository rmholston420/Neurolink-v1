"""Unit tests for neurolink.dsp.artifact_gate.

Public API confirmed from source:
  ArtifactGate(config)
  .evaluate(eeg, accel=None) -> ArtifactDecision   # NO fs parameter
  .get_stats() / .reset_stats() / .get_config() / .set_config()

ArtifactDecision fields:
  .reject  (bool)  — True when the frame is contaminated
  .reasons (list[str])
  .clean   (property) — returns `not self.reject`
"""
from __future__ import annotations

import numpy as np
import pytest

from neurolink.dsp.artifact_gate import ArtifactDecision, ArtifactGate, GateConfig
from neurolink.dsp.artifact_config import ARTIFACT_PK2PK_UV, ARTIFACT_ACCEL_RMS_G

N_SAMPLES = 256
N_CH = 4


@pytest.fixture
def gate() -> ArtifactGate:
    cfg = GateConfig(pk2pk_uv=100.0, electrode_type="wet")
    return ArtifactGate(config=cfg)


@pytest.fixture
def clean_eeg() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(0, 5.0, (N_CH, N_SAMPLES))


@pytest.fixture
def high_amplitude_eeg() -> np.ndarray:
    """EEG with pk2pk >> 100 uV on all channels."""
    eeg = np.zeros((N_CH, N_SAMPLES))
    eeg[:, 0] = 200.0
    eeg[:, -1] = -200.0
    return eeg


@pytest.fixture
def motion_accel() -> np.ndarray:
    rng = np.random.default_rng(9)
    return rng.normal(0, 1.0, (3, N_SAMPLES))  # 1 g RMS >> 0.15 g threshold


class TestConstruction:
    def test_default_construction(self):
        g = ArtifactGate()
        assert g is not None

    def test_custom_config(self):
        cfg = GateConfig(pk2pk_uv=50.0, electrode_type="dry")
        g = ArtifactGate(config=cfg)
        assert g.get_config().pk2pk_uv == pytest.approx(50.0)


class TestCleanEEG:
    def test_clean_frame_passes(self, gate, clean_eeg):
        decision = gate.evaluate(clean_eeg)
        assert isinstance(decision, ArtifactDecision)
        assert decision.clean is True

    def test_clean_frame_no_reject_reasons(self, gate, clean_eeg):
        decision = gate.evaluate(clean_eeg)
        assert decision.reject is False


class TestAmplitudeGate:
    def test_high_amplitude_rejected(self, gate, high_amplitude_eeg):
        decision = gate.evaluate(high_amplitude_eeg)
        assert decision.reject is True

    def test_high_amplitude_has_reason(self, gate, high_amplitude_eeg):
        decision = gate.evaluate(high_amplitude_eeg)
        assert len(decision.reasons) >= 1


class TestMotionGate:
    def test_motion_rejects(self, gate, clean_eeg, motion_accel):
        decision = gate.evaluate(clean_eeg, accel=motion_accel)
        assert decision.reject is True

    def test_no_accel_does_not_reject(self, gate, clean_eeg):
        decision = gate.evaluate(clean_eeg, accel=None)
        assert decision.clean is True


class TestStats:
    def test_stats_structure(self, gate, clean_eeg):
        gate.evaluate(clean_eeg)
        stats = gate.get_stats()
        assert "total_frames" in stats
        assert stats["total_frames"] >= 1

    def test_reset_stats(self, gate, clean_eeg):
        gate.evaluate(clean_eeg)
        gate.reset_stats()
        assert gate.get_stats()["total_frames"] == 0


class TestEdgeCases:
    def test_too_few_samples(self, gate):
        eeg = np.zeros((4, 1))
        decision = gate.evaluate(eeg)
        assert isinstance(decision, ArtifactDecision)

    def test_5ch_eeg_no_crash(self, gate):
        eeg = np.random.default_rng(3).normal(0, 5.0, (5, N_SAMPLES))
        decision = gate.evaluate(eeg)
        assert isinstance(decision, ArtifactDecision)

    def test_none_eeg_returns_clean(self, gate):
        decision = gate.evaluate(None)  # type: ignore[arg-type]
        assert decision.clean is True
