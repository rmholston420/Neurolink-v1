"""Unit tests for dsp/decoders.py."""
from __future__ import annotations

from neurolink.dsp.decoders import decode_eeg, decode_ppg, decode_imu


def test_decode_eeg_returns_empty_for_short_packet():
    assert decode_eeg(bytes(10)) == []


def test_decode_eeg_returns_12_samples_for_valid_packet():
    """A valid 20-byte packet should return 12 float samples."""
    data = bytes(20)  # all zeros
    result = decode_eeg(data)
    assert len(result) == 12
    assert all(isinstance(v, float) for v in result)


def test_decode_ppg_returns_empty_for_short_packet():
    assert decode_ppg(bytes(10)) == []


def test_decode_ppg_returns_6_samples_for_valid_packet():
    data = bytes(20)
    result = decode_ppg(data)
    assert len(result) == 6


def test_decode_imu_returns_empty_for_short_packet():
    accel, gyro = decode_imu(bytes(10))
    assert accel == []
    assert gyro == []


def test_decode_imu_returns_9_values_for_valid_packet():
    data = bytes(20)
    accel, gyro = decode_imu(data)
    assert len(accel) == 9
    assert len(gyro) == 9
