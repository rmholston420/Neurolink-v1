"""Unit tests for dsp/artifact_config.py — constant values and types."""

from __future__ import annotations

import neurolink.dsp.artifact_config as cfg


class TestArtifactConfigTypes:
    """Every exported name must be the declared type."""

    def test_all_floats_are_float(self):
        float_names = [
            "ARTIFACT_PK2PK_UV",
            "ARTIFACT_ACCEL_RMS_G",
            "ARTIFACT_KURTOSIS_THRESHOLD",
            "ARTIFACT_BLINK_FRONTAL_UV",
            "BLINK_FREQ_HZ_MAX",
            "BLINK_LOW_FREQ_RATIO_MIN",
            "BLINK_FRONTAL_RATIO",
            "ARTIFACT_HEOG_ASYMMETRY_UV",
            "HEOG_FREQ_HZ_MAX",
            "ARTIFACT_EMG_HF_RATIO",
            "EMG_FREQ_LOW_HZ",
            "EMG_FREQ_HIGH_HZ",
            "ARTIFACT_LINE_FREQ_HZ",
            "ARTIFACT_LINE_BAND_HZ",
            "ARTIFACT_LINE_POWER_RATIO",
            "CARDIAC_FREQ_LOW_HZ",
            "CARDIAC_FREQ_HIGH_HZ",
            "CARDIAC_TEMPORAL_UV",
            "ELECTRODE_POP_STEP_UV",
            "ELECTRODE_POP_ISOLATION_RATIO",
            "ASR_BURST_SD",
            "ASR_CALIB_SEC",
            "BASELINE_TOTAL_SEC",
            "BASELINE_DISCARD_SEC",
            "EA1_ALPHA_THRESHOLD",
            "EA1_THETA_THRESHOLD",
            "EA1_CONTACT_QUALITY_MIN",
        ]
        for name in float_names:
            val = getattr(cfg, name)
            assert isinstance(val, float), f"{name} should be float, got {type(val)}"


class TestArtifactConfigValues:
    """Key constants must sit within clinically sensible bounds."""

    # Amplitude gate
    def test_pk2pk_uv_positive(self):
        assert cfg.ARTIFACT_PK2PK_UV > 0

    def test_pk2pk_uv_clinical_range(self):
        """Valid EEG pk2pk threshold should be between 50 and 300 µV."""
        assert 50.0 <= cfg.ARTIFACT_PK2PK_UV <= 300.0

    # IMU motion gate
    def test_accel_rms_g_positive(self):
        assert cfg.ARTIFACT_ACCEL_RMS_G > 0

    def test_accel_rms_g_subg(self):
        """Should be a sub-g fraction (quiet resting head)."""
        assert cfg.ARTIFACT_ACCEL_RMS_G < 1.0

    # Kurtosis
    def test_kurtosis_threshold_positive(self):
        assert cfg.ARTIFACT_KURTOSIS_THRESHOLD > 0

    # Blink
    def test_blink_frontal_uv_positive(self):
        assert cfg.ARTIFACT_BLINK_FRONTAL_UV > 0

    def test_blink_frontal_uv_less_than_pk2pk(self):
        """Blink threshold should be sub-rejection-gate."""
        assert cfg.ARTIFACT_BLINK_FRONTAL_UV < cfg.ARTIFACT_PK2PK_UV

    def test_blink_freq_max_in_eeg_band(self):
        assert 0 < cfg.BLINK_FREQ_HZ_MAX <= 20.0

    def test_blink_low_freq_ratio_0_to_1(self):
        assert 0.0 < cfg.BLINK_LOW_FREQ_RATIO_MIN <= 1.0

    def test_blink_frontal_ratio_gt_1(self):
        assert cfg.BLINK_FRONTAL_RATIO > 1.0

    # HEOG
    def test_heog_asymmetry_positive(self):
        assert cfg.ARTIFACT_HEOG_ASYMMETRY_UV > 0

    def test_heog_freq_max_low(self):
        assert cfg.HEOG_FREQ_HZ_MAX <= 10.0

    # EMG
    def test_emg_hf_ratio_0_to_1(self):
        assert 0.0 < cfg.ARTIFACT_EMG_HF_RATIO < 1.0

    def test_emg_band_ordering(self):
        assert cfg.EMG_FREQ_LOW_HZ < cfg.EMG_FREQ_HIGH_HZ

    # Line noise
    def test_line_freq_valid(self):
        assert cfg.ARTIFACT_LINE_FREQ_HZ in (50.0, 60.0)

    def test_line_band_hz_positive(self):
        assert cfg.ARTIFACT_LINE_BAND_HZ > 0

    def test_line_power_ratio_0_to_1(self):
        assert 0.0 < cfg.ARTIFACT_LINE_POWER_RATIO < 1.0

    # Cardiac
    def test_cardiac_band_ordering(self):
        assert cfg.CARDIAC_FREQ_LOW_HZ < cfg.CARDIAC_FREQ_HIGH_HZ

    def test_cardiac_freq_low_in_delta(self):
        assert cfg.CARDIAC_FREQ_LOW_HZ >= 0.5

    def test_cardiac_temporal_uv_positive(self):
        assert cfg.CARDIAC_TEMPORAL_UV > 0

    # Electrode pop
    def test_pop_step_uv_positive(self):
        assert cfg.ELECTRODE_POP_STEP_UV > 0

    def test_pop_isolation_ratio_gt_1(self):
        assert cfg.ELECTRODE_POP_ISOLATION_RATIO > 1.0

    # ASR
    def test_asr_burst_sd_gt_zero(self):
        assert cfg.ASR_BURST_SD > 0

    def test_asr_calib_sec_positive(self):
        assert cfg.ASR_CALIB_SEC > 0

    # Baseline
    def test_baseline_discard_less_than_total(self):
        assert cfg.BASELINE_DISCARD_SEC < cfg.BASELINE_TOTAL_SEC

    def test_baseline_total_sec_gte_150(self):
        assert cfg.BASELINE_TOTAL_SEC >= 150.0

    def test_baseline_discard_sec_gte_30(self):
        assert cfg.BASELINE_DISCARD_SEC >= 30.0

    # EA-1 thresholds
    def test_ea1_thresholds_in_unit_interval(self):
        for name in ("EA1_ALPHA_THRESHOLD", "EA1_THETA_THRESHOLD", "EA1_CONTACT_QUALITY_MIN"):
            val = getattr(cfg, name)
            assert 0.0 < val <= 1.0, f"{name}={val} not in (0, 1]"


class TestArtifactConfigInternalConsistency:
    """Cross-constant consistency."""

    def test_emg_detection_band_above_beta(self):
        """EMG is high-frequency; low bound should be above beta (30 Hz)."""
        assert cfg.EMG_FREQ_LOW_HZ >= 30.0

    def test_cardiac_and_blink_do_not_overlap(self):
        """Cardiac band (0.8–1.8 Hz) and blink band (0–10 Hz) overlap is fine,
        but cardiac high should be below blink freq max."""
        assert cfg.CARDIAC_FREQ_HIGH_HZ < cfg.BLINK_FREQ_HZ_MAX

    def test_heog_band_below_blink_band(self):
        """HEOG saccades are lower frequency than blinks."""
        assert cfg.HEOG_FREQ_HZ_MAX <= cfg.BLINK_FREQ_HZ_MAX
