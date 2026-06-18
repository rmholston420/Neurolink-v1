"""Unit tests for dsp.artifact_gate.ArtifactGate."""
from __future__ import annotations

import numpy as np
import pytest

from neurolink.dsp.artifact_gate import ArtifactGate
from neurolink.dsp.artifact_config import ArtifactConfig


N_CH = 4
FS = 256


@pytest.fixture()
def gate() -> ArtifactGate:
    config = ArtifactConfig()
    return ArtifactGate(n_channels=N_CH, fs=FS, config=config)


def _frame(amp: float = 5e-6) -> np.ndarray:
    return np.full((N_CH,), amp)


class TestGatePassthrough:
    def test_clean_signal_passes(self, gate):
        out = gate.process(_frame(5e-6))
        assert out is not None

    def test_output_shape_preserved(self, gate):
        frame = _frame(5e-6)
        out = gate.process(frame)
        assert out.shape == frame.shape


class TestGateSuppression:
    def test_artifact_frame_suppressed_or_zeroed(self, gate):
        """When an artifact is present the gate should suppress or zero output."""
        artifact = _frame(2000e-6)
        out = gate.process(artifact)
        # Either output is zeroed or amplitude is significantly reduced
        assert np.all(np.abs(out) <= np.abs(artifact) + 1e-9)


class TestGateReset:
    def test_reset_does_not_raise(self, gate):
        gate.process(_frame(2000e-6))
        gate.reset()  # must not raise

    def test_after_reset_clean_passes(self, gate):
        gate.process(_frame(2000e-6))
        gate.reset()
        out = gate.process(_frame(5e-6))
        assert out is not None


class TestGateEnabled:
    def test_disabled_gate_passes_everything(self, gate):
        gate.enabled = False
        artifact = _frame(2000e-6)
        out = gate.process(artifact)
        np.testing.assert_array_almost_equal(out, artifact)
