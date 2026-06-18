"""Targeted tests for dsp/fnirs.py missing lines.

Missing lines:
  117-125  - channel-count mismatch guard: cached baseline has wrong n_ch
  128-133  - spike clip path (rm/rm2/nf populated)
  139      - baseline detrend: bl is None on first call -> initialised from frame mean
  155-159  - Welford update on subsequent frames
  173      - decode: n_ch < min_channels -> empty arrays
  182-184  - decode: Beer-Lambert loop over n_pairs
"""

from __future__ import annotations

import numpy as np
import pytest

import neurolink.dsp.fnirs as fnirs


@pytest.fixture(autouse=True)
def _reset_fnirs_state():
    """Isolate each test: reset module-level mutable state before and after."""
    fnirs.reset()
    yield
    fnirs.reset()


def _frame(n_ch: int = 4, n_samples: int = 16, value: float = 1.0) -> np.ndarray:
    return np.full((n_ch, n_samples), value, dtype=np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# apply() — basic paths
# ═════════════════════════════════════════════════════════════════════════════

class TestFNIRSApplyBasic:
    def test_none_input_returns_none(self):
        assert fnirs.apply(None) is None

    def test_disabled_returns_input_unchanged(self):
        fnirs.set_config(enable=False)
        raw = _frame()
        out = fnirs.apply(raw)
        assert out is raw

    def test_1d_array_returned_unchanged(self):
        """Non-2D array falls through the ndim guard."""
        raw = np.ones(8, dtype=np.float32)
        out = fnirs.apply(raw)
        assert out is raw

    def test_zero_channel_array_returned_unchanged(self):
        raw = np.zeros((0, 16), dtype=np.float32)
        out = fnirs.apply(raw)
        assert out is raw

    def test_returns_float32_copy(self):
        """Output must be a float32 copy — never mutates input."""
        raw = _frame(value=2.0)
        out = fnirs.apply(raw)
        assert out is not raw
        assert out.dtype == np.float32


# ═════════════════════════════════════════════════════════════════════════════
# apply() — line 139: baseline initialised from first frame  
# ═════════════════════════════════════════════════════════════════════════════

class TestFNIRSApplyFirstFrame:
    def test_first_frame_initialises_baseline(self):
        """Line 139: bl is None -> baseline set from frame mean. Module state updated."""
        assert fnirs._baseline is None
        fnirs.apply(_frame(value=2.0))
        assert fnirs._baseline is not None
        assert fnirs._n_frames == 1
        assert fnirs._running_mean is not None

    def test_first_frame_output_near_zero_mean(self):
        """After detrending the first frame, channel means should be near 0."""
        raw = _frame(value=3.0)
        out = fnirs.apply(raw)
        # The detrend subtracts approx the frame mean; result ~= 0
        assert np.abs(out.mean()) < 0.5


# ═════════════════════════════════════════════════════════════════════════════
# apply() — lines 155-159: Welford online update on frame 2+
# ═════════════════════════════════════════════════════════════════════════════

class TestFNIRSApplyWelford:
    def test_n_frames_increments_each_call(self):
        for i in range(1, 6):
            fnirs.apply(_frame())
            assert fnirs._n_frames == i

    def test_running_mean_updates(self):
        fnirs.apply(_frame(value=1.0))
        mean_after_1 = fnirs._running_mean.copy()
        fnirs.apply(_frame(value=3.0))
        # Running mean should have moved toward 3.0
        assert not np.allclose(fnirs._running_mean, mean_after_1)

    def test_running_m2_non_negative(self):
        for _ in range(5):
            fnirs.apply(_frame(value=float(np.random.rand())))
        assert np.all(fnirs._running_m2 >= 0)


# ═════════════════════════════════════════════════════════════════════════════
# apply() — lines 128-133: spike clip path (nf > 1, rm/rm2 populated)
# ═════════════════════════════════════════════════════════════════════════════

class TestFNIRSApplySpikeClip:
    def test_spike_clip_reduces_outlier(self):
        """Lines 128-133: after 2+ frames, spikes beyond threshold are clipped."""
        fnirs.set_config(spike_threshold=1.0)  # tight threshold
        # Warm up with 2 normal frames
        for _ in range(2):
            fnirs.apply(_frame(value=1.0))
        # Spike frame: value far outside the running window
        spike_raw = _frame(value=1000.0)
        out = fnirs.apply(spike_raw)
        # Spike should be clipped — output mean must be much less than 1000
        assert float(out.mean()) < 100.0

    def test_normal_frame_passes_through_spike_clip_unchanged(self):
        """Within threshold, spike clip does not distort the signal."""
        fnirs.set_config(spike_threshold=10.0)
        for _ in range(3):
            fnirs.apply(_frame(value=1.0))
        out = fnirs.apply(_frame(value=1.05))  # tiny deviation
        # Output should still be near zero after detrending
        assert np.abs(out.mean()) < 1.0


# ═════════════════════════════════════════════════════════════════════════════
# apply() — lines 117-125: channel-count mismatch guard
# ═════════════════════════════════════════════════════════════════════════════

class TestFNIRSChannelMismatch:
    def test_channel_mismatch_resets_state_and_processes_new_frame(self):
        """
        Lines 117-125: if _baseline has a different channel count than the
        incoming frame, the cached state must be discarded and re-initialised.
        """
        # Prime state with 4-channel data
        fnirs.apply(_frame(n_ch=4, value=1.0))
        assert fnirs._n_frames == 1
        assert fnirs._baseline.shape[0] == 4

        # Now send an 8-channel frame -> mismatch triggers guard
        out = fnirs.apply(_frame(n_ch=8, value=1.0))
        # State should have been reset and re-initialised for 8 channels
        assert fnirs._baseline.shape[0] == 8
        assert out is not None
        assert out.shape[0] == 8

    def test_channel_mismatch_resets_n_frames_to_1(self):
        fnirs.apply(_frame(n_ch=4))
        fnirs.apply(_frame(n_ch=4))
        assert fnirs._n_frames == 2
        # Mismatch: switch to 6 channels
        fnirs.apply(_frame(n_ch=6))
        # State was wiped; only the new frame has been processed
        assert fnirs._n_frames == 1


# ═════════════════════════════════════════════════════════════════════════════
# decode() — lines 173, 182-184
# ═════════════════════════════════════════════════════════════════════════════

class TestFNIRSDecode:
    def test_none_returns_none(self):
        assert fnirs.decode(None) is None

    def test_1d_array_returned_unchanged(self):
        raw = np.ones(8, dtype=np.float32)
        out = fnirs.decode(raw)
        assert out is raw

    def test_fewer_channels_than_min_returns_empty_arrays(self):
        """Line 173: n_ch < min_channels -> tuple of empty arrays."""
        fnirs.set_config(min_channels=4)
        raw = np.ones((2, 16), dtype=np.float32)  # 2 < 4
        result = fnirs.decode(raw)
        assert isinstance(result, tuple)
        hbo, hbr = result
        assert hbo.shape[0] == 0
        assert hbr.shape[0] == 0

    def test_single_channel_below_min_2_returns_empty(self):
        """n_ch < 2 always returns empty regardless of config."""
        fnirs.set_config(min_channels=1)
        raw = np.ones((1, 16), dtype=np.float32)
        hbo, hbr = fnirs.decode(raw)
        assert hbo.shape[0] == 0

    def test_valid_4ch_frame_returns_hbo_hbr_arrays(self):
        """Lines 182-184: Beer-Lambert loop produces HbO/HbR for each pair."""
        raw = np.ones((4, 32), dtype=np.float32) * 0.1
        result = fnirs.decode(raw)
        assert isinstance(result, tuple)
        hbo, hbr = result
        # 4 channels -> 2 pairs
        assert hbo.shape == (2, 32)
        assert hbr.shape == (2, 32)
        assert hbo.dtype == np.float32
        assert hbr.dtype == np.float32

    def test_hbo_hbr_values_are_finite(self):
        raw = (np.random.rand(4, 64) * 0.5).astype(np.float32)
        hbo, hbr = fnirs.decode(raw)
        assert np.all(np.isfinite(hbo))
        assert np.all(np.isfinite(hbr))

    def test_6ch_produces_3_pairs(self):
        raw = np.ones((6, 16), dtype=np.float32) * 0.2
        hbo, hbr = fnirs.decode(raw)
        assert hbo.shape[0] == 3
        assert hbr.shape[0] == 3


# ══════════════════════════════════════════════════════════════════════════───
# config helpers
# ══════════════════════════════════════════════════════════════════════════───

class TestFNIRSConfig:
    def test_get_config_returns_copy(self):
        cfg1 = fnirs.get_config()
        cfg2 = fnirs.get_config()
        assert cfg1 is not cfg2

    def test_set_config_updates_field(self):
        fnirs.set_config(baseline_alpha=0.05)
        assert fnirs.get_config().baseline_alpha == pytest.approx(0.05)

    def test_set_config_ignores_invalid_key(self):
        original = fnirs.get_config().baseline_alpha
        fnirs.set_config(nonexistent_key=999)
        assert fnirs.get_config().baseline_alpha == pytest.approx(original)

    def test_reset_clears_baseline(self):
        fnirs.apply(_frame())
        assert fnirs._baseline is not None
        fnirs.reset()
        assert fnirs._baseline is None
        assert fnirs._n_frames == 0
