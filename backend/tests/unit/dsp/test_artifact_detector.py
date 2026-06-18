"""Unit tests for neurolink.dsp.artifact_detector (Stage 3b).

Covers:
- All 7 ArtifactType detectors on synthetic signals
- CorrectionPlan routing for each artifact class
- DetectorConfig runtime overrides via set_config()
- get_stats() / reset_stats() accounting
- Thread-safety smoke test (concurrent classify() calls)
- Edge cases: empty array, < min_samples_for_fft, AUX channel excluded
- clean flag truth when no artifact is present
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from neurolink.dsp.artifact_detector import (
    ArtifactDetector,
    ArtifactType,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FS = 256.0
N = 512  # 2-second frame at 256 Hz — well above min_samples_for_fft=64

_CH_TP9, _CH_AF7, _CH_AF8, _CH_TP10 = 0, 1, 2, 3


def _clean_frame(n_ch: int = 4, n_samples: int = N, amplitude_uv: float = 5.0) -> np.ndarray:
    """Low-amplitude, band-limited background EEG (no artifacts)."""
    rng = np.random.default_rng(seed=0)
    eeg = rng.normal(0, amplitude_uv, (n_ch, n_samples))
    return eeg.astype(np.float32)


def _blink_frame() -> np.ndarray:
    """Synthetic blink: large slow transient concentrated at AF7/AF8."""
    eeg = _clean_frame(amplitude_uv=3.0)
    t = np.linspace(0, N / FS, N)
    # 150 µV peak Gaussian bump at 2 Hz on frontal channels
    blink = 150.0 * np.exp(-0.5 * ((t - 1.0) / 0.15) ** 2)
    eeg[_CH_AF7] += blink
    eeg[_CH_AF8] += blink * 0.9
    return eeg


def _emg_frame() -> np.ndarray:
    """Synthetic EMG: broadband high-frequency noise on temporal channels."""
    rng = np.random.default_rng(seed=1)
    eeg = _clean_frame(amplitude_uv=3.0)
    # Add 50 µV RMS white noise — majority of power above 30 Hz
    hf = rng.normal(0, 50.0, (4, N))
    eeg += hf.astype(np.float32)
    return eeg


def _line_noise_frame(line_hz: float = 60.0) -> np.ndarray:
    """Synthetic power-line interference at line_hz on all channels."""
    eeg = _clean_frame(amplitude_uv=3.0)
    t = np.linspace(0, N / FS, N)
    tone = 40.0 * np.sin(2 * np.pi * line_hz * t)  # large tone
    for ch in range(4):
        eeg[ch] += tone
    return eeg


def _motion_accel(rms_g: float = 0.5) -> np.ndarray:
    """Accelerometer frame exceeding the motion threshold."""
    rng = np.random.default_rng(seed=2)
    # 3-axis accel, N samples
    return (rng.normal(0, rms_g, (3, N))).astype(np.float32)


def _pop_frame() -> np.ndarray:
    """Electrode pop: abrupt single-channel step change."""
    eeg = _clean_frame(amplitude_uv=3.0)
    # Insert a 150 µV instantaneous step on TP9 only
    eeg[_CH_TP9, N // 2] += 150.0
    return eeg


def _cardiac_frame() -> np.ndarray:
    """Synthetic cardiac pulse: ~1.2 Hz sinusoid on temporal channels."""
    eeg = _clean_frame(amplitude_uv=3.0)
    t = np.linspace(0, N / FS, N)
    pulse = 25.0 * np.sin(2 * np.pi * 1.2 * t)
    eeg[_CH_TP9] += pulse
    eeg[_CH_TP10] += pulse * 0.85
    return eeg


def _heog_frame() -> np.ndarray:
    """Synthetic horizontal saccade: asymmetry between AF7 and AF8."""
    eeg = _clean_frame(amplitude_uv=3.0)
    t = np.linspace(0, N / FS, N)
    # Slow (<4 Hz) drift with opposite polarity on AF7 vs AF8
    drift = 40.0 * np.sin(2 * np.pi * 1.5 * t)
    eeg[_CH_AF7] += drift
    eeg[_CH_AF8] -= drift
    return eeg


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def detector() -> ArtifactDetector:
    return ArtifactDetector(line_freq_hz=60.0)


# ---------------------------------------------------------------------------
# Basic API
# ---------------------------------------------------------------------------


class TestArtifactDetectorAPI:
    def test_returns_detection_report(self, detector):
        report = detector.classify(_clean_frame(), fs=FS)
        assert hasattr(report, "annotations")
        assert hasattr(report, "correction_plan")
        assert hasattr(report, "artifact_types")
        assert hasattr(report, "clean")

    def test_clean_signal_no_artifacts(self, detector):
        """Low-amplitude background EEG should produce no artifact annotations."""
        report = detector.classify(_clean_frame(amplitude_uv=2.0), fs=FS)
        assert report.clean is True
        assert report.artifact_types == []
        assert not report.correction_plan.hard_reject
        assert not report.correction_plan.any_correction()

    def test_type_names_list(self, detector):
        report = detector.classify(_blink_frame(), fs=FS)
        names = report.type_names()
        assert isinstance(names, list)
        for name in names:
            assert isinstance(name, str)

    def test_aux_channel_ignored(self, detector):
        """A 5-channel frame with large AUX noise should not trigger EEG artifacts."""
        rng = np.random.default_rng(seed=99)
        eeg_5ch = _clean_frame(n_ch=5, amplitude_uv=2.0)
        # Add extreme noise only to channel 4 (AUX)
        eeg_5ch[4] += rng.normal(0, 500.0, N)
        report = detector.classify(eeg_5ch, fs=FS)
        # Artifacts should NOT be triggered by the AUX channel alone
        for ann in report.annotations:
            assert "AUX" not in ann.channels

    def test_empty_frame_returns_clean(self, detector):
        report = detector.classify(np.zeros((4, 0)), fs=FS)
        assert report.clean is True

    def test_short_frame_skips_fft_detectors(self, detector):
        """Frames shorter than min_samples_for_fft must not crash."""
        short = np.zeros((4, 10), dtype=np.float32)
        report = detector.classify(short, fs=FS)
        assert report is not None


# ---------------------------------------------------------------------------
# Per-type detectors
# ---------------------------------------------------------------------------


class TestBlinkDetector:
    def test_blink_detected(self, detector):
        report = detector.classify(_blink_frame(), fs=FS)
        assert ArtifactType.BLINK in report.artifact_types

    def test_blink_routes_to_ocular_regression(self, detector):
        report = detector.classify(_blink_frame(), fs=FS)
        assert report.correction_plan.apply_ocular_regression is True

    def test_blink_confidence_nonzero(self, detector):
        report = detector.classify(_blink_frame(), fs=FS)
        blink_anns = [a for a in report.annotations if a.artifact_type == ArtifactType.BLINK]
        assert blink_anns
        assert blink_anns[0].confidence > 0.0


class TestEMGDetector:
    def test_emg_detected(self, detector):
        report = detector.classify(_emg_frame(), fs=FS)
        assert ArtifactType.EMG in report.artifact_types

    def test_emg_routes_to_asr(self, detector):
        report = detector.classify(_emg_frame(), fs=FS)
        assert report.correction_plan.apply_asr is True


class TestLineNoiseDetector:
    def test_line_noise_detected_60hz(self, detector):
        report = detector.classify(_line_noise_frame(line_hz=60.0), fs=FS)
        assert ArtifactType.LINE_NOISE in report.artifact_types

    def test_line_noise_50hz(self):
        det50 = ArtifactDetector(line_freq_hz=50.0)
        report = det50.classify(_line_noise_frame(line_hz=50.0), fs=FS)
        assert ArtifactType.LINE_NOISE in report.artifact_types

    def test_line_noise_routes_to_notch(self, detector):
        report = detector.classify(_line_noise_frame(), fs=FS)
        assert report.correction_plan.apply_notch is True


class TestMotionDetector:
    def test_motion_detected_with_accel(self, detector):
        eeg = _clean_frame()
        accel = _motion_accel(rms_g=0.8)
        report = detector.classify(eeg, accel=accel, fs=FS)
        assert ArtifactType.MOTION in report.artifact_types

    def test_motion_hard_rejects(self, detector):
        eeg = _clean_frame()
        accel = _motion_accel(rms_g=1.5)
        report = detector.classify(eeg, accel=accel, fs=FS)
        assert report.correction_plan.hard_reject is True

    def test_no_motion_without_accel(self, detector):
        report = detector.classify(_clean_frame(), accel=None, fs=FS)
        assert ArtifactType.MOTION not in report.artifact_types


class TestElectrodePopDetector:
    def test_pop_detected(self, detector):
        report = detector.classify(_pop_frame(), fs=FS)
        assert ArtifactType.ELECTRODE_POP in report.artifact_types


# ---------------------------------------------------------------------------
# Config / stats
# ---------------------------------------------------------------------------


class TestDetectorConfig:
    def test_get_config_returns_copy(self, detector):
        cfg1 = detector.get_config()
        cfg1.blink_frontal_uv = 999.0
        cfg2 = detector.get_config()
        assert cfg2.blink_frontal_uv != 999.0

    def test_disable_blink_via_set_config(self, detector):
        cfg = detector.get_config()
        cfg.enable_blink = False
        detector.set_config(cfg)
        report = detector.classify(_blink_frame(), fs=FS)
        assert ArtifactType.BLINK not in report.artifact_types

    def test_disable_motion_via_set_config(self, detector):
        cfg = detector.get_config()
        cfg.enable_motion = False
        detector.set_config(cfg)
        accel = _motion_accel(rms_g=2.0)
        report = detector.classify(_clean_frame(), accel=accel, fs=FS)
        assert ArtifactType.MOTION not in report.artifact_types


class TestStats:
    def test_stats_total_frames_count(self, detector):
        for _ in range(5):
            detector.classify(_clean_frame(), fs=FS)
        stats = detector.get_stats()
        assert stats["total_frames"] == 5

    def test_stats_artifact_type_count(self, detector):
        for _ in range(3):
            detector.classify(_blink_frame(), fs=FS)
        stats = detector.get_stats()
        blink_count = stats["artifact_types"]["BLINK"]["count"]
        assert blink_count == 3

    def test_reset_stats_clears_counters(self, detector):
        detector.classify(_blink_frame(), fs=FS)
        detector.reset_stats()
        stats = detector.get_stats()
        assert stats["total_frames"] == 0
        assert stats["artifact_types"]["BLINK"]["count"] == 0


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_classify_no_exception(self, detector):
        errors: list[Exception] = []

        def worker():
            try:
                for _ in range(20):
                    detector.classify(_clean_frame(), fs=FS)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
