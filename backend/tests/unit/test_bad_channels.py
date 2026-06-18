"""Unit tests for dsp.bad_channels.BadChannelDetector."""
from __future__ import annotations

import numpy as np
import pytest

from neurolink.dsp.bad_channels import BadChannelDetector


N_CH = 4
FS = 256


@pytest.fixture()
def detector() -> BadChannelDetector:
    return BadChannelDetector(n_channels=N_CH, fs=FS)


def _good_data(n_samples: int = FS * 5) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(0, 10e-6, size=(n_samples, N_CH))


def _data_with_flat(ch_idx: int = 1) -> np.ndarray:
    data = _good_data()
    data[:, ch_idx] = 0.0  # flat-line
    return data


def _data_with_high_amp(ch_idx: int = 2) -> np.ndarray:
    data = _good_data()
    data[:, ch_idx] = 1000e-6  # constant high amplitude
    return data


class TestConstruction:
    def test_instantiation(self):
        bcd = BadChannelDetector(n_channels=N_CH, fs=FS)
        assert bcd is not None

    def test_channel_count(self, detector):
        assert detector.n_channels == N_CH


class TestDetection:
    def test_all_good_channels_returns_empty_mask(self, detector):
        mask = detector.detect(_good_data())
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (N_CH,)
        assert not np.any(mask)  # no bad channels

    def test_flat_channel_detected(self, detector):
        mask = detector.detect(_data_with_flat(ch_idx=1))
        assert mask[1]  # channel 1 should be flagged

    def test_good_channels_not_flagged(self, detector):
        mask = detector.detect(_data_with_flat(ch_idx=1))
        for i in range(N_CH):
            if i != 1:
                assert not mask[i]

    def test_high_amplitude_channel_detected(self, detector):
        mask = detector.detect(_data_with_high_amp(ch_idx=2))
        assert mask[2]


class TestMaskReturnType:
    def test_mask_is_bool_array(self, detector):
        mask = detector.detect(_good_data())
        assert mask.dtype == bool


class TestReset:
    def test_reset_does_not_raise(self, detector):
        detector.detect(_good_data())
        detector.reset()

    def test_after_reset_detects_correctly(self, detector):
        detector.detect(_data_with_flat())
        detector.reset()
        mask = detector.detect(_good_data())
        assert not np.any(mask)
