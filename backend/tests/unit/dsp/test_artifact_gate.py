"""Unit tests for dsp/artifact_gate.py — Stage 3 artifact gate."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from neurolink.dsp.artifact_config import (
    ARTIFACT_ACCEL_RMS_G,
    ARTIFACT_KURTOSIS_THRESHOLD,
    ARTIFACT_PK2PK_UV,
)
from neurolink.dsp.artifact_gate import (
    _ELECTRODE_PK2PK_DEFAULTS,
    ArtifactDecision,
    ArtifactGate,
    GateConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_eeg(
    n_ch: int = 4,
    n_samples: int = 256,
    amplitude: float = 5.0,
    seed: int = 0,
) -> np.ndarray:
    """Return low-amplitude Gaussian EEG unlikely to trigger any gate."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n_ch, n_samples)) * amplitude).astype(np.float32)


def _gate_with_explicit_config(
    pk2pk_uv: float = ARTIFACT_PK2PK_UV,
    accel_rms_g: float = ARTIFACT_ACCEL_RMS_G,
    kurtosis_threshold: float = ARTIFACT_KURTOSIS_THRESHOLD,
    electrode_type: str = "wet",
) -> ArtifactGate:
    """Bypass adapter_factory auto-detection by injecting an explicit config."""
    cfg = GateConfig(
        electrode_type=electrode_type,
        pk2pk_uv=pk2pk_uv,
        accel_rms_g=accel_rms_g,
        kurtosis_threshold=kurtosis_threshold,
    )
    return ArtifactGate(config=cfg)


def _dynamic_accel(n_samples: int = 64, amplitude: float = 5.0) -> np.ndarray:
    """Return a (3, n_samples) sinusoidal accel array with high AC-RMS.

    AC-RMS = amplitude / sqrt(2) ≈ 3.54 g for amplitude=5.0, which is
    well above the default 0.15 g threshold.  Using a sine wave guarantees
    the mean-subtracted (AC) component is non-zero, so the IMU gate fires.
    """
    t = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    wave = (amplitude * np.sin(t)).astype(np.float32)
    return np.stack([wave, wave, wave], axis=0)  # shape (3, n_samples)


# ---------------------------------------------------------------------------
# GateConfig
# ---------------------------------------------------------------------------


class TestGateConfig:
    def test_explicit_electrode_type_preserved(self):
        cfg = GateConfig(electrode_type="wet", pk2pk_uv=100.0)
        assert cfg.electrode_type == "wet"
        assert cfg.pk2pk_uv == pytest.approx(100.0)

    def test_dry_pk2pk_tighter_than_wet(self):
        assert _ELECTRODE_PK2PK_DEFAULTS["dry"] < _ELECTRODE_PK2PK_DEFAULTS["wet"]

    def test_semi_pk2pk_between_dry_and_wet(self):
        assert (
            _ELECTRODE_PK2PK_DEFAULTS["dry"]
            < _ELECTRODE_PK2PK_DEFAULTS["semi"]
            < _ELECTRODE_PK2PK_DEFAULTS["wet"]
        )

    def test_defaults_from_artifact_config(self):
        cfg = GateConfig(electrode_type="wet", pk2pk_uv=None)
        assert cfg.pk2pk_uv == pytest.approx(ARTIFACT_PK2PK_UV)


# ---------------------------------------------------------------------------
# ArtifactDecision
# ---------------------------------------------------------------------------


class TestArtifactDecision:
    def test_default_is_clean(self):
        d = ArtifactDecision()
        assert d.clean is True
        assert d.reject is False
        assert d.reasons == []

    def test_add_reason_sets_reject(self):
        d = ArtifactDecision()
        d.add_reason("test_reason")
        assert d.reject is True
        assert d.clean is False
        assert "test_reason" in d.reasons

    def test_multiple_reasons(self):
        d = ArtifactDecision()
        d.add_reason("a")
        d.add_reason("b")
        assert len(d.reasons) == 2


# ---------------------------------------------------------------------------
# ArtifactGate.evaluate() — guards
# ---------------------------------------------------------------------------


class TestArtifactGateGuards:
    def test_none_eeg_returns_clean(self):
        gate = _gate_with_explicit_config()
        result = gate.evaluate(None)
        assert result.clean is True

    def test_1d_eeg_returns_clean(self):
        gate = _gate_with_explicit_config()
        result = gate.evaluate(np.zeros(256, dtype=np.float32))
        assert result.clean is True

    def test_single_sample_returns_clean(self):
        gate = _gate_with_explicit_config()
        result = gate.evaluate(np.zeros((4, 1), dtype=np.float32))
        assert result.clean is True

    def test_5ch_eeg_only_evaluates_first_4(self):
        """AUX channel (index 4) must not trigger amplitude gate."""
        gate = _gate_with_explicit_config(pk2pk_uv=10.0)
        eeg = np.zeros((5, 256), dtype=np.float32)
        eeg[4, :] = 500.0
        result = gate.evaluate(eeg)
        assert result.clean is True


# ---------------------------------------------------------------------------
# Amplitude gate
# ---------------------------------------------------------------------------


class TestAmplitudeGate:
    def test_clean_signal_passes(self):
        gate = _gate_with_explicit_config(pk2pk_uv=100.0)
        eeg = _clean_eeg(amplitude=5.0)
        result = gate.evaluate(eeg)
        assert result.clean is True

    def test_over_threshold_rejected(self):
        gate = _gate_with_explicit_config(pk2pk_uv=50.0)
        eeg = np.zeros((4, 256), dtype=np.float32)
        eeg[0, 0] = 200.0
        result = gate.evaluate(eeg)
        assert result.reject is True
        assert any("amplitude" in r for r in result.reasons)

    def test_reason_contains_electrode_type(self):
        gate = _gate_with_explicit_config(pk2pk_uv=50.0, electrode_type="dry")
        eeg = np.zeros((4, 256), dtype=np.float32)
        eeg[1, 0] = 300.0
        result = gate.evaluate(eeg)
        assert any("dry" in r for r in result.reasons)

    def test_exactly_at_threshold_passes(self):
        """pk2pk == threshold is not > threshold; should pass.

        Kurtosis is disabled so a single-spike signal does not
        confound the amplitude boundary test.
        """
        cfg = GateConfig(
            electrode_type="wet",
            pk2pk_uv=100.0,
            enable_kurtosis=False,
        )
        gate = ArtifactGate(config=cfg)
        eeg = np.zeros((4, 256), dtype=np.float32)
        eeg[0, 0] = 100.0  # pk2pk = 100.0 == threshold → passes (not >)
        result = gate.evaluate(eeg)
        assert result.clean is True

    def test_disabled_amplitude_gate_passes_large_signal(self):
        """When amplitude gate is disabled, even 9999 µV must pass.

        Kurtosis is also disabled so the spike does not trigger
        a different gate and confound the result.
        """
        cfg = GateConfig(
            electrode_type="wet",
            pk2pk_uv=10.0,
            enable_amplitude=False,
            enable_kurtosis=False,
        )
        gate = ArtifactGate(config=cfg)
        eeg = np.zeros((4, 256), dtype=np.float32)
        eeg[0, 0] = 9999.0
        result = gate.evaluate(eeg)
        assert result.clean is True


# ---------------------------------------------------------------------------
# IMU gate
# ---------------------------------------------------------------------------


class TestIMUGate:
    def test_low_accel_passes(self):
        gate = _gate_with_explicit_config(accel_rms_g=0.15)
        accel = np.ones((3, 64), dtype=np.float32) * 0.05
        result = gate.evaluate(_clean_eeg(), accel=accel)
        assert result.clean is True

    def test_high_accel_rejected(self):
        """Dynamic (AC) motion above threshold must be rejected.

        The IMU gate uses AC-RMS (mean-subtracted) to ignore steady-state
        gravity.  A sinusoidal signal has AC-RMS = amplitude/sqrt(2) which
        is far above the 0.15 g threshold when amplitude=5.0.
        """
        gate = _gate_with_explicit_config(accel_rms_g=0.15)
        accel = _dynamic_accel(n_samples=64, amplitude=5.0)
        result = gate.evaluate(_clean_eeg(), accel=accel)
        assert result.reject is True
        assert any("imu" in r for r in result.reasons)

    def test_no_accel_skips_imu_gate(self):
        gate = _gate_with_explicit_config()
        result = gate.evaluate(_clean_eeg(), accel=None)
        assert not any("imu" in r for r in result.reasons)

    def test_1d_accel_accepted(self):
        gate = _gate_with_explicit_config(accel_rms_g=0.15)
        accel = np.ones(64, dtype=np.float32) * 0.05
        result = gate.evaluate(_clean_eeg(), accel=accel)
        assert result.clean is True

    def test_disabled_imu_gate_ignores_motion(self):
        cfg = GateConfig(electrode_type="wet", pk2pk_uv=100.0, accel_rms_g=0.01, enable_imu=False)
        gate = ArtifactGate(config=cfg)
        accel = _dynamic_accel(n_samples=64, amplitude=10.0)
        result = gate.evaluate(_clean_eeg(), accel=accel)
        assert not any("imu" in r for r in result.reasons)


# ---------------------------------------------------------------------------
# Kurtosis gate
# ---------------------------------------------------------------------------


class TestKurtosisGate:
    def test_gaussian_signal_passes(self):
        gate = _gate_with_explicit_config(kurtosis_threshold=5.0)
        rng = np.random.default_rng(42)
        eeg = rng.standard_normal((4, 512)).astype(np.float32)
        result = gate.evaluate(eeg)
        assert not any("kurtosis" in r for r in result.reasons)

    def test_spike_causes_high_kurtosis(self):
        gate = _gate_with_explicit_config(kurtosis_threshold=3.0)
        eeg = np.zeros((4, 512), dtype=np.float32)
        eeg[0, 256] = 1000.0
        result = gate.evaluate(eeg)
        assert any("kurtosis" in r for r in result.reasons)

    def test_disabled_kurtosis_gate(self):
        cfg = GateConfig(
            electrode_type="wet", pk2pk_uv=500.0, kurtosis_threshold=0.0001, enable_kurtosis=False
        )
        gate = ArtifactGate(config=cfg)
        eeg = np.zeros((4, 256), dtype=np.float32)
        eeg[0, 128] = 9999.0
        result = gate.evaluate(eeg)
        assert not any("kurtosis" in r for r in result.reasons)


# ---------------------------------------------------------------------------
# Multiple simultaneous flags
# ---------------------------------------------------------------------------


class TestMultipleReasons:
    def test_amplitude_and_imu_both_flagged(self):
        """Both amplitude and IMU gates must fire independently.

        Uses a sinusoidal accel signal so AC-RMS >> threshold (0.01 g).
        A constant accel (np.ones * 5.0) would have AC-RMS = 0 after
        mean subtraction and would NOT trigger the IMU gate.
        """
        gate = _gate_with_explicit_config(pk2pk_uv=10.0, accel_rms_g=0.01)
        eeg = np.zeros((4, 256), dtype=np.float32)
        eeg[0, 0] = 500.0
        accel = _dynamic_accel(n_samples=64, amplitude=5.0)
        result = gate.evaluate(eeg, accel=accel)
        assert result.reject is True
        assert any("amplitude" in r for r in result.reasons)
        assert any("imu" in r for r in result.reasons)


# ---------------------------------------------------------------------------
# get_stats / reset_stats
# ---------------------------------------------------------------------------


class TestArtifactGateStats:
    def test_initial_stats_zero(self):
        gate = _gate_with_explicit_config()
        s = gate.get_stats()
        assert s["total_frames"] == 0
        assert s["rejected_frames"] == 0
        assert s["rejection_rate"] == 0.0

    def test_clean_frame_increments_total(self):
        gate = _gate_with_explicit_config()
        gate.evaluate(_clean_eeg())
        assert gate.get_stats()["total_frames"] == 1
        assert gate.get_stats()["rejected_frames"] == 0

    def test_rejected_frame_increments_both(self):
        gate = _gate_with_explicit_config(pk2pk_uv=10.0)
        eeg = np.zeros((4, 256), dtype=np.float32)
        eeg[0, 0] = 500.0
        gate.evaluate(eeg)
        s = gate.get_stats()
        assert s["total_frames"] == 1
        assert s["rejected_frames"] == 1
        assert s["rejection_rate"] == pytest.approx(1.0)

    def test_rejection_rate_fraction(self):
        gate = _gate_with_explicit_config(pk2pk_uv=10.0)
        clean = _clean_eeg(amplitude=1.0)
        dirty = np.zeros((4, 256), dtype=np.float32)
        dirty[0, 0] = 500.0
        gate.evaluate(clean)
        gate.evaluate(dirty)
        s = gate.get_stats()
        assert s["rejection_rate"] == pytest.approx(0.5)

    def test_reset_stats_clears_counts(self):
        gate = _gate_with_explicit_config()
        gate.evaluate(_clean_eeg())
        gate.reset_stats()
        s = gate.get_stats()
        assert s["total_frames"] == 0
        assert s["rejected_frames"] == 0

    def test_none_eeg_does_not_increment_total(self):
        gate = _gate_with_explicit_config()
        gate.evaluate(None)
        assert gate.get_stats()["total_frames"] == 0


# ---------------------------------------------------------------------------
# get_config / set_config
# ---------------------------------------------------------------------------


class TestArtifactGateConfig:
    def test_get_config_returns_copy(self):
        gate = _gate_with_explicit_config()
        c1 = gate.get_config()
        c2 = gate.get_config()
        assert c1 is not c2

    def test_set_config_updates_pk2pk(self):
        gate = _gate_with_explicit_config(pk2pk_uv=100.0)
        gate.set_config(GateConfig(electrode_type="wet", pk2pk_uv=50.0))
        assert gate.get_config().pk2pk_uv == pytest.approx(50.0)

    def test_set_config_live_effect(self):
        """Frame that was clean under old config is rejected after set_config.

        Both configs have kurtosis disabled to isolate the amplitude gate.
        """
        cfg_loose = GateConfig(electrode_type="wet", pk2pk_uv=200.0, enable_kurtosis=False)
        gate = ArtifactGate(config=cfg_loose)
        eeg = np.zeros((4, 256), dtype=np.float32)
        eeg[0, 0] = 120.0  # pk2pk=120, clean under 200 threshold
        assert gate.evaluate(eeg).clean is True

        gate.set_config(GateConfig(electrode_type="wet", pk2pk_uv=80.0, enable_kurtosis=False))
        assert gate.evaluate(eeg).reject is True


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestArtifactGateThreadSafety:
    def test_concurrent_evaluate_does_not_raise(self):
        gate = _gate_with_explicit_config()
        errors: list[Exception] = []

        def worker():
            try:
                for _ in range(30):
                    gate.evaluate(_clean_eeg())
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []

    def test_concurrent_set_config_and_evaluate(self):
        gate = _gate_with_explicit_config()
        errors: list[Exception] = []

        def evaluator():
            try:
                for _ in range(30):
                    gate.evaluate(_clean_eeg())
            except Exception as exc:
                errors.append(exc)

        def configurator():
            try:
                for _ in range(10):
                    gate.set_config(GateConfig(electrode_type="wet", pk2pk_uv=100.0))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=evaluator) for _ in range(3)] + [
            threading.Thread(target=configurator)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
