"""Unit tests for neurolink.dsp.artifact_detector.

All imports and fixture values are derived from the module's actual
public API (ArtifactDetector, DetectorConfig, ArtifactType,
DetectionReport, CorrectionPlan, ArtifactAnnotation).

artifact_config.py exports only module-level constants — there is no
ArtifactConfig class.  Tests that need threshold values import the
constants directly.
"""
from __future__ import annotations

import numpy as np
import pytest

from neurolink.dsp.artifact_detector import (
    ArtifactAnnotation,
    ArtifactDetector,
    ArtifactType,
    CorrectionPlan,
    DetectionReport,
    DetectorConfig,
)
from neurolink.dsp.artifact_config import (
    ARTIFACT_BLINK_FRONTAL_UV,
    ARTIFACT_ACCEL_RMS_G,
    ARTIFACT_EMG_HF_RATIO,
)

FS = 256.0
N_SAMPLES = 256  # 1 second of data at 256 Hz
N_CH = 4  # TP9, AF7, AF8, TP10


@pytest.fixture
def detector() -> ArtifactDetector:
    return ArtifactDetector()


@pytest.fixture
def clean_eeg() -> np.ndarray:
    """4-channel clean EEG: low-amplitude Gaussian noise."""
    rng = np.random.default_rng(42)
    return rng.normal(0, 5.0, (N_CH, N_SAMPLES))  # 5 uV RMS


@pytest.fixture
def blink_eeg() -> np.ndarray:
    """Synthetic blink: large slow transient at AF7/AF8 (ch 1,2)."""
    rng = np.random.default_rng(0)
    eeg = rng.normal(0, 3.0, (N_CH, N_SAMPLES))
    # Add 150 uV slow sine at 1 Hz on frontal channels
    t = np.linspace(0, 1, N_SAMPLES)
    blink = 150.0 * np.sin(2 * np.pi * 1.0 * t)
    eeg[1] += blink  # AF7
    eeg[2] += blink  # AF8
    return eeg


@pytest.fixture
def motion_accel() -> np.ndarray:
    """High RMS accelerometer (AC component >> threshold)."""
    rng = np.random.default_rng(7)
    accel = rng.normal(0, 1.0, (3, N_SAMPLES))  # 1 g RMS >> 0.15 g threshold
    return accel


@pytest.fixture
def quiet_accel() -> np.ndarray:
    """Near-zero AC accelerometer (below motion threshold)."""
    rng = np.random.default_rng(8)
    return rng.normal(0, 0.01, (3, N_SAMPLES))  # 0.01 g RMS


class TestConstruction:
    def test_default_construction(self):
        d = ArtifactDetector()
        assert d is not None

    def test_custom_config(self):
        cfg = DetectorConfig(blink_frontal_uv=60.0)
        d = ArtifactDetector(config=cfg)
        assert d.get_config().blink_frontal_uv == 60.0

    def test_custom_line_freq(self):
        d = ArtifactDetector(line_freq_hz=50.0)
        assert d.get_config().line_freq_hz == 50.0


class TestCleanEEG:
    def test_clean_frame_returns_clean(self, detector, clean_eeg):
        report = detector.classify(clean_eeg, fs=FS)
        assert report.clean is True
        assert report.annotations == []
        assert report.artifact_types == []

    def test_clean_frame_no_hard_reject(self, detector, clean_eeg):
        report = detector.classify(clean_eeg, fs=FS)
        assert report.correction_plan.hard_reject is False

    def test_clean_frame_no_corrections(self, detector, clean_eeg):
        plan = detector.classify(clean_eeg, fs=FS).correction_plan
        assert not plan.any_correction()


class TestMotionDetection:
    def test_motion_sets_hard_reject(self, detector, clean_eeg, motion_accel):
        report = detector.classify(clean_eeg, accel=motion_accel, fs=FS)
        assert report.correction_plan.hard_reject is True

    def test_motion_type_in_report(self, detector, clean_eeg, motion_accel):
        report = detector.classify(clean_eeg, accel=motion_accel, fs=FS)
        assert ArtifactType.MOTION in report.artifact_types

    def test_quiet_accel_no_motion(self, detector, clean_eeg, quiet_accel):
        report = detector.classify(clean_eeg, accel=quiet_accel, fs=FS)
        assert ArtifactType.MOTION not in report.artifact_types

    def test_no_accel_no_motion(self, detector, clean_eeg):
        report = detector.classify(clean_eeg, accel=None, fs=FS)
        assert ArtifactType.MOTION not in report.artifact_types


class TestBlinkDetection:
    def test_blink_detected(self, detector, blink_eeg):
        report = detector.classify(blink_eeg, fs=FS)
        assert ArtifactType.BLINK in report.artifact_types

    def test_blink_triggers_ocular_regression(self, detector, blink_eeg):
        report = detector.classify(blink_eeg, fs=FS)
        assert report.correction_plan.apply_ocular_regression is True

    def test_blink_not_hard_rejected(self, detector, blink_eeg):
        report = detector.classify(blink_eeg, fs=FS)
        assert report.correction_plan.hard_reject is False

    def test_blink_annotation_fields(self, detector, blink_eeg):
        report = detector.classify(blink_eeg, fs=FS)
        blink_anns = [a for a in report.annotations if a.artifact_type == ArtifactType.BLINK]
        assert len(blink_anns) >= 1
        ann = blink_anns[0]
        assert 0.0 <= ann.confidence <= 1.0
        assert ann.feature_value > 0
        assert ann.threshold == pytest.approx(ARTIFACT_BLINK_FRONTAL_UV)


class TestStats:
    def test_stats_count_increments(self, detector, clean_eeg):
        before = detector.get_stats()["total_frames"]
        detector.classify(clean_eeg, fs=FS)
        after = detector.get_stats()["total_frames"]
        assert after == before + 1

    def test_reset_stats(self, detector, clean_eeg):
        detector.classify(clean_eeg, fs=FS)
        detector.reset_stats()
        assert detector.get_stats()["total_frames"] == 0

    def test_stats_has_all_artifact_types(self, detector):
        stats = detector.get_stats()
        for art_type in ArtifactType:
            assert art_type.name in stats["artifact_types"]


class TestConfigUpdate:
    def test_set_config_takes_effect(self, detector):
        cfg = DetectorConfig(enable_blink=False)
        detector.set_config(cfg)
        assert detector.get_config().enable_blink is False

    def test_get_config_returns_copy(self, detector):
        c1 = detector.get_config()
        c1.blink_frontal_uv = 9999.0
        assert detector.get_config().blink_frontal_uv != 9999.0


class TestEdgeCases:
    def test_too_few_samples_returns_clean(self, detector):
        eeg = np.zeros((4, 1))
        report = detector.classify(eeg, fs=FS)
        assert report.clean is True

    def test_none_eeg_returns_clean(self, detector):
        report = detector.classify(None, fs=FS)  # type: ignore[arg-type]
        assert report.clean is True

    def test_5_channel_eeg_no_crash(self, detector):
        eeg = np.random.default_rng(1).normal(0, 5.0, (5, N_SAMPLES))
        report = detector.classify(eeg, fs=FS)
        assert isinstance(report, DetectionReport)

    def test_correction_plan_any_correction_false_when_clean(self):
        plan = CorrectionPlan()
        assert not plan.any_correction()

    def test_correction_plan_any_correction_true_when_notch(self):
        plan = CorrectionPlan(apply_notch=True)
        assert plan.any_correction()
