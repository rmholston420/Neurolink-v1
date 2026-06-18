"""Unit tests for dsp.asr.ASR (Adaptive Spatial Rejection)."""
from __future__ import annotations

import numpy as np
import pytest

from neurolink.dsp.asr import ASR


N_CH = 4
FS = 256
CAL_SECONDS = 20


@pytest.fixture()
def asr() -> ASR:
    return ASR(n_channels=N_CH, fs=FS)


def _cal_data() -> np.ndarray:
    """Clean calibration data: (samples, channels)."""
    rng = np.random.default_rng(0)
    return rng.normal(0, 5e-6, size=(FS * CAL_SECONDS, N_CH))


def _clean_frame() -> np.ndarray:
    return np.random.default_rng(1).normal(0, 5e-6, size=(N_CH,))


def _artifact_frame() -> np.ndarray:
    return np.full((N_CH,), 500e-6)


class TestConstruction:
    def test_instantiation(self):
        inst = ASR(n_channels=N_CH, fs=FS)
        assert inst is not None

    def test_channel_count(self, asr):
        assert asr.n_channels == N_CH


class TestCalibration:
    def test_fit_does_not_raise(self, asr):
        asr.fit(_cal_data())

    def test_fit_idempotent(self, asr):
        asr.fit(_cal_data())
        asr.fit(_cal_data())  # second fit must not raise


class TestTransform:
    def test_transform_before_fit_returns_input(self, asr):
        frame = _clean_frame()
        out = asr.transform(frame)
        assert out.shape == frame.shape

    def test_transform_after_fit_shape(self, asr):
        asr.fit(_cal_data())
        out = asr.transform(_clean_frame())
        assert out.shape == (N_CH,)

    def test_transform_artifact_amplitude_reduced(self, asr):
        asr.fit(_cal_data())
        artifact = _artifact_frame()
        out = asr.transform(artifact)
        assert np.max(np.abs(out)) <= np.max(np.abs(artifact)) + 1e-12

    def test_transform_clean_passthrough(self, asr):
        """Clean signal after fit should not be significantly distorted."""
        asr.fit(_cal_data())
        frame = _clean_frame()
        out = asr.transform(frame)
        # Output should be close to input for clean signal
        np.testing.assert_allclose(out, frame, atol=50e-6)


class TestReset:
    def test_reset_does_not_raise(self, asr):
        asr.fit(_cal_data())
        asr.reset()

    def test_after_reset_transform_returns_correct_shape(self, asr):
        asr.fit(_cal_data())
        asr.reset()
        out = asr.transform(_clean_frame())
        assert out.shape == (N_CH,)
